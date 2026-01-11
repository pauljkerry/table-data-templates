import gc
import os
import re
from dataclasses import dataclass, field
from time import perf_counter as now
from typing import Optional

import rmm
import cupy as cp
import numpy as np
import polars as pl
import rmm.mr as mr
from rmm.allocators.cupy import rmm_cupy_allocator
from sklearn.metrics import mean_squared_error

from src.utils.loggers import CVLogger, NoOpLogger
from src.utils.print_duration import print_duration
from src.utils.mem_info import free_ram_gib, free_vram_gib
from abc import ABC, abstractmethod


@dataclass
class TrainResult:
    model: any
    val_pred: np.ndarray | None = None
    evals_result: dict | None = None
    extra: dict | None = None
    fi: pl.DataFrame | None = None
    best_iteration: int | None = None


# 共通の親クラス
@dataclass
class BaseCVTrainer(ABC):
    data_id: str
    train_paths: str | list[str]
    test_paths: str | list[str] | None = None
    features: Optional[list[str]] = None
    target: str = "target"
    fold_name: Optional[str] = None
    weight_col: Optional[str] = None
    cat_cols: Optional[list[str]] = None

    params: dict = field(default_factory=dict)

    seed: int = 42
    gpu: bool = True

    opts: dict = field(init=True, default_factory=dict)

    def __post_init__(self):
        if isinstance(self.train_paths, (str, os.PathLike)):
            self.train_paths = [str(self.train_paths)]
        else:
            self.train_paths = [str(p) for p in self.train_paths]

        if self.test_paths:
            if isinstance(self.test_paths, (str, os.PathLike)):
                self.test_paths = [str(self.test_paths)]
            else:
                self.test_paths = [str(p) for p in self.test_paths]

        self.lf_train = pl.scan_parquet(self.train_paths)
        self.lf_test = (
            pl.scan_parquet(self.test_paths) if self.test_paths else None
        )

        self.rep_metric = "rmse"
        self.metrics = {
            "rmse": lambda y, t: np.sqrt(mean_squared_error(y, t)),
        }
        hdr = pl.read_parquet(self.train_paths, n_rows=0)
        self.all_cols = pl.read_parquet(self.train_paths, n_rows=0).columns

        self.fold_df = (
             pl.read_parquet(f"../../artifacts/folds/{self.fold_name}.parquet")
             .sort("row_id")  # 一応ソート
        )
        self.unique_folds = np.sort(self.fold_df["fold"].unique().to_numpy())
        self.n_folds = len(self.unique_folds)

        print(
            f"Loaded folds data. Detected {self.n_folds} "
            f"folds: {self.unique_folds}"
        )

        if self.cat_cols is None:
            self.cat_cols = [
                c for c, dt in zip(hdr.columns, hdr.dtypes)
                if dt == pl.Categorical
            ]

        if self.features is None:
            meta = {
                c
                for c in ("row_id", self.target, self.weight_col)
                if c and c in self.all_cols
            }

            self.features = [
                c for c in self.all_cols
                if c not in meta
            ]

        dev_mr = mr.CudaAsyncMemoryResource()
        mr.set_current_device_resource(dev_mr)
        rmm.reinitialize(
            managed_memory=False,
            initial_pool_size=None,
        )
        cp.cuda.set_allocator(rmm_cupy_allocator)

        cp.get_default_memory_pool().set_limit(4 * 1024**3)
        self.pmp = cp.cuda.PinnedMemoryPool()
        cp.cuda.set_pinned_memory_allocator(self.pmp.malloc)

    # --- 共通の処理フロー (fit) ---
    def fit(
        self,
        loggers: list[CVLogger] | None = None,
        one_fold: bool = False,
        full_train: bool = False
    ) -> dict:
        t_total_start = now()

        loggers = loggers or [NoOpLogger()]
        meta = {
            "data_id": self.data_id,
            "seed": self.seed,
            "fold_name": self.fold_name,
            **self.params,
            **self.opts
        }
        for lg in loggers:
            lg.on_start(meta)

        if not one_fold:
            train_rows = (
                self.lf_train
                .select(pl.len())
                .collect()
                .item()
            )
            test_rows = (
                pl.scan_parquet(self.test_paths)
                  .select(pl.len())
                  .collect()
                  .item()
            )

            oof = np.zeros(train_rows, dtype=np.float32)
            test_pred = np.zeros(test_rows, dtype=np.float32)

        fold_scores = {name: [] for name in self.metrics.keys()}
        fi_list = []

        if full_train:
            return self._run_full_train(loggers, t_total_start)

        for i, fold in enumerate(self.unique_folds):
            title = f" Fold {i+1} / {self.n_folds} "
            print("=" * 48)
            print(f"{title:=^48}")
            print("=" * 48)
            print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
            print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")
            t_fold_start = now()
            fold_summary = {}

            val_idx = (
                self.fold_df
                .filter(pl.col("fold") == fold)
                .get_column("row_id")
                .to_list()
            )

            train_result: TrainResult = self.train_model(fold)

            fi_list.append(train_result.fi)
            model = train_result.model
            val_pred = train_result.val_pred
            if train_result.best_iteration is not None:
                fold_summary[f"best_{self.log_axis_name}"] = train_result.best_iteration

            y_valid = (
                pl.read_parquet(
                    self.train_paths, columns=[self.target, "row_id"]
                )
                .filter(pl.col("row_id").is_in(val_idx))
                .sort("row_id")
                .select(self.target)
                .to_numpy()
                .astype(np.float32)
                .ravel()
            )

            for name, metric_func in self.metrics.items():
                val_score = metric_func(y_valid, val_pred)
                print(f"{name.upper()} Valid: {val_score:.5f}")
                fold_summary[name] = val_score
                fold_scores[name].append(val_score)

            fold_summary["runtime"] = print_duration(
                t_fold_start, now(), f"Fold {fold} Runtime"
            )

            for lg in loggers:
                lg.on_fold_end(
                    fold,
                    axis_name=self.log_axis_name,
                    evals_result=train_result.evals_result,
                    extra=train_result.extra,
                    summary=fold_summary
                )

            if one_fold:
                result = {
                    "oof": None,
                    "test_pred": None,
                    "oof_score": fold_scores[self.rep_metric][0]
                }
            else:
                oof[val_idx] = val_pred
                test_pred += self.predict_test(model)

            del model, train_result
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            self.pmp.free_all_blocks()

            if one_fold:
                return result

        # --- Result集計 ---
        y = (
            pl.read_parquet(self.train_paths, columns=self.target)
              .get_column(self.target)
              .cast(pl.Float32)
              .to_numpy()
        )
        test_pred /= self.n_folds

        oofs = {
            name: metric_func(y, oof)
            for name, metric_func in self.metrics.items()
        }

        oof_stats = {
            name: {
                "oof": oofs[name],
                "mean": np.mean(vals),
                "std": np.std(vals)
            }
            for name, vals in fold_scores.items()
        }

        print(f"\n{' CV Results ':*^48}")
        print("─" * 48)
        print(f" {'Metric':^9}  {'OOF':>10}  {'Mean':>10} ± {'Std':<10} ")
        print("-" * 48)
        for name, stats in oof_stats.items():
            print(f" {name.upper():^9} "
                  f" {stats['oof']:>10.5f} "
                  f" {stats['mean']:>10.5f} ± {stats['std']:<10.5f} ")
        print("─" * 48)

        print(f"Free CPU Mem: {round(free_ram_gib(), 2)} GB")
        print(f"Free GPU Mem: {round(free_vram_gib(), 2)} GB")

        result = {
            "oof": oof,
            "test_pred": test_pred,
            "oof_score": oofs[self.rep_metric]
        }

        # Feature Importanceの記録
        valid_fi = [df for df in fi_list if df is not None]
        if valid_fi:
            all_fi = pl.concat(valid_fi, how="vertical_relaxed")
            fi_mean = (
                all_fi
                .group_by("Feature")
                .agg([
                    pl.sum("Importance").alias("Importance")
                ])
            ).sort("Importance", descending=True)

            result["fi_mean"] = fi_mean

        overall_summary = {}
        for name, stats in oof_stats.items():
            overall_summary[f"{name}_mean"] = stats["mean"]
            overall_summary[f"{name}_std"] = stats["std"]
            overall_summary[f"{name}_oof"] = oofs[name]

        overall_summary["total_runtime"] = print_duration(
            t_total_start, now(), "Total CV Runtime"
        )

        for lg in loggers:
            lg.on_end(overall_summary)

        return result

    @abstractmethod
    def train_model(
        self,
        fold: int
    ) -> tuple[any, np.ndarray, np.ndarray, dict | None, dict | None]:
        pass

    @abstractmethod
    def predict_test(
        self,
        model: any
    ) -> np.ndarray:
        pass

    def _run_full_train(self, loggers, t_start):
        print("=" * 48)
        print(f"{' FULL TRAIN START ':^48}")
        print("=" * 48)

        result: TrainResult = self.train_on_all_data()

        test_pred = self.predict_test(result.model)

        total_time = print_duration(t_start, now(), "Full Train Runtime")

        summary = {"total_runtime": total_time}
        if result.best_iteration:
            summary[f"best_{self.log_axis_name}"] = result.best_iteration

        for lg in loggers:
            lg.on_end(summary)

        return {
            "oof": None,
            "test_pred": test_pred,
            "oof_score": None,
        }

    @abstractmethod
    def train_on_all_data(self) -> TrainResult:
        pass
