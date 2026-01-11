from __future__ import annotations
import gc
import re
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, List

import cudf
import cupy as cp
import polars as pl
import pyarrow.parquet as pq
import xgboost as xgb
from src.models.base_cv_trainer import BaseCVTrainer, TrainResult


@dataclass(eq=False)
class ParquetIter(xgb.core.DataIter):
    paths: list[str]

    features: list[str] = None
    target: str = "target"
    cat_cols: Optional[Iterable[str]] = None
    folds: cp.array = None
    include_fold: str = None
    exclude_fold: str = None
    weight_col: Optional[str] = None

    rowgroup_batch: int = 1
    gpu: Optional[bool] = None
    predict_mode: bool = False

    # === 内部状態（initの引数にしないもの） ===
    _temporary_data: Any = field(init=False, default=None, repr=False)
    _pass_count: int = field(init=False, default=0, repr=False)

    _files: List[dict] = field(init=False, default_factory=list, repr=False)
    _file_idx: int = field(init=False, default=0, repr=False)
    _rg_idx: int = field(init=False, default=0, repr=False)
    _columns: List[str] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self) -> None:
        super().__init__()
        hdr = pl.read_parquet(self.paths, n_rows=0)
        all_cols = hdr.columns

        # 入力列（重複除去）
        cols = list(self.features)
        if (not self.predict_mode) and (self.target in all_cols):
            cols.append(self.target)
        if (not self.predict_mode) and (self.weight_col in all_cols):
            cols.append(self.weight_col)
        self._columns = list(dict.fromkeys(cols))
        if "row_id" not in self._columns:
            self._columns.append("row_id")

        # 内部状態
        self._reader = None
        self._current_file_index = 0

        # 各ファイルの row group 数を先に調べておく
        self._files = []
        for p in self.paths:
            pf = pq.ParquetFile(p)
            self._files.append({"path": p, "nrg": pf.num_row_groups})

        self._file_idx = 0
        self._rg_idx = 0

    def reset(self):
        self._file_idx = 0
        self._rg_idx = 0
        self._pass_count += 1

    def next(self, input_data):
        """
        1 回呼ばれるごとに row group の束 (rowgroup_batch) を 1 塊だけ返す。
        """
        while True:
            if self._file_idx >= len(self._files):
                return 0  # 終了

            rec = self._files[self._file_idx]
            path, nrg = rec["path"], rec["nrg"]

            if self._rg_idx >= nrg:
                # 次のファイルへ
                self._file_idx += 1
                self._rg_idx = 0
                continue

            # このバッチで読む row groups
            start = self._rg_idx
            end = min(self._rg_idx + self.rowgroup_batch, nrg)
            bundle = list(range(start, end))
            self._rg_idx = end  # 次に備える

            # === ここがコア：cuDF で row group を直接読む ===
            # ※ cudf.read_parquet は単一ファイル向け。複数パスは「ループで回す」方針。
            gdf = cudf.read_parquet(path, columns=self._columns, row_groups=bundle)

            if self.folds is not None:
                # fold_df (row_id, fold) をマージ
                gdf = gdf.merge(self.folds, on="row_id", how="inner")
                gdf = gdf.sort_values("row_id")

                # マージした fold 列を使ってフィルタリング
                if self.include_fold is not None:
                    gdf = gdf[gdf["fold"] == self.include_fold]
                if self.exclude_fold is not None:
                    gdf = gdf[gdf["fold"] != self.exclude_fold]

                # 学習に使わないので fold 列と row_id (特徴量に含まれない場合) を削除
                # ※ row_id が特徴量に含まれるなら残す
                drop_cols = ["fold"]
                if "row_id" not in self.features:
                    drop_cols.append("row_id")
                gdf.drop(columns=drop_cols, inplace=True, errors="ignore")

            # カテゴリ化（必要な列のみ）
            if self.cat_cols:
                for c in self.cat_cols:
                    gdf[c] = gdf[c].astype("category")

            # 出力
            if self.predict_mode:
                input_data(data=gdf[self.features])
            else:
                kwargs = dict(data=gdf[self.features], label=gdf[self.target])
                if self.weight_col and self.weight_col in gdf.columns:
                    kwargs["weight"] = gdf[self.weight_col]
                input_data(**kwargs)

            # 後始末（参照を断つ→GC）
            del gdf
            gc.collect()
            return 1


@dataclass
class XGBCVTrainer(BaseCVTrainer):
    def __post_init__(self):
        super().__post_init__()
        self.log_axis_name = "iter"

        default_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "learning_rate": 0.1,
            "max_depth": 7,
            "min_child_weight": 10.0,
            "gamma": 0,
            "colsample_bytree": 0.4,
            "subsample": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "verbosity": 0,
            "tree_method": "hist",
            "device": "cuda",
            "seed": self.seed,
            "max_bin": 256,
            "grow_policy": "depthwise",
            "predictor": "gpu_predictor"
        }

        user_params = self.params or {}
        merged = {**default_params, **user_params}

        self.early_stopping_rounds = self.opts.get(
            "early_stopping_rounds",
            None
        )
        self.num_boost_round = self.opts.get(
            "num_boost_round",
            20000
        )

        # ユーザー未指定なら lr に応じて自動設定（下限あり）
        if self.early_stopping_rounds is None:
            lr = float(merged["learning_rate"])
            self.early_stopping_rounds = max(50, int(math.ceil(10.0 / lr)))

        # train() の引数として取り出す
        self.params = merged

        self.fold_cudf = cudf.DataFrame(
            self.fold_df.select(["row_id", "fold"]).to_numpy(),
            columns=["row_id", "fold"]
        )

    def train_model(self, fold) -> TrainResult:
        train_it = ParquetIter(
            paths=self.train_paths,
            features=self.features,
            target=self.target,
            cat_cols=self.cat_cols,
            exclude_fold=fold,
            folds=self.fold_cudf,
            weight_col=self.weight_col,
            gpu=True
        )
        valid_it = ParquetIter(
            paths=self.train_paths,
            features=self.features,
            target=self.target,
            cat_cols=self.cat_cols,
            include_fold=fold,
            folds=self.fold_cudf,
            weight_col=self.weight_col,
            gpu=self.gpu
        )

        self.dtrain = xgb.QuantileDMatrix(
            train_it,
            enable_categorical=True
        )
        dvalid = xgb.QuantileDMatrix(
            valid_it,
            enable_categorical=True,
            ref=self.dtrain
        )

        evals_result = {}

        model = xgb.train(
            self.params,
            self.dtrain,
            num_boost_round=self.num_boost_round,
            evals=[(self.dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=100,
            evals_result=evals_result,
        )

        importances = model.get_score(importance_type="total_gain")
        total_gain = float(sum(importances.values()))
        fi_df = pl.DataFrame(
            {
                "Feature": list(importances.keys()),
                "Importance": [
                    ((v/total_gain)*100.0)/self.n_folds
                    for v in importances.values()
                ],
            }
        )
        val_pred = model.predict(
            dvalid, iteration_range=(0, model.best_iteration + 1)
        )
        # GPU配列(CuPy)ならCPU(NumPy)に戻す
        if hasattr(val_pred, "get"):
            val_pred = val_pred.get()
        elif hasattr(val_pred, "to_numpy"):
            val_pred = val_pred.to_numpy()

        return TrainResult(
                    model=model,
                    val_pred=val_pred,
                    evals_result=evals_result,
                    fi=fi_df,
                    best_iteration=model.best_iteration
                )

    def predict_test(self, model):
        test_it = ParquetIter(
            paths=self.test_paths,
            features=self.features,
            target=self.target,
            cat_cols=self.cat_cols,
            predict_mode=True,
            gpu=self.gpu
        )
        dtest = xgb.QuantileDMatrix(
            test_it,
            enable_categorical=True,
            ref=self.dtrain
        )
        pred = model.predict(
            dtest,
            iteration_range=(0, model.best_iteration + 1)
        )

        if hasattr(pred, "get"):
            pred = pred.get()
        return pred

    def train_on_all_data(self) -> TrainResult:
        boost_rounds = int(
            self.num_boost_round * (self.n_folds/(self.n_folds-1))
        )

        train_it = ParquetIter(
            paths=self.train_paths,
            features=self.features,
            target=self.target,
            cat_cols=self.cat_cols,
            weight_col=self.weight_col,
            gpu=self.gpu
        )

        self.dtrain = xgb.QuantileDMatrix(
            train_it,
            enable_categorical=True
        )

        model = xgb.train(
            self.params,
            self.dtrain,
            num_boost_round=boost_rounds,
            evals=[]
        )

        return TrainResult(
            model=model,
            val_pred=None,
            evals_result=None,
            extra=None,
            fi=None,
            best_iteration=model.best_iteration
        )
