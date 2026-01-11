from dataclasses import dataclass

import cudf
import cupy as cp
import polars as pl
from cuml.linear_model import Ridge

from src.models.base_cv_trainer import BaseCVTrainer, TrainResult
from src.utils.compute_feature_stats import compute_feature_stats


@dataclass
class RidgeCVTrainer(BaseCVTrainer):
    def __post_init__(self):
        super().__post_init__()
        self.log_axis_name = "iter"

        default_params = {
            "alpha": 1.0,
            "fit_intercept": True,
            "solver": "auto"
        }
        self.params = {**default_params, **self.params}

        # Cat colsを除外
        self.features = [c for c in self.features if c not in self.cat_cols]

        self.mean, self.std = compute_feature_stats(
            self.train_paths,
            self.features,
            self.features,
        )
        self.mean = cp.asarray(self.mean, dtype=cp.float32)
        self.std = cp.asarray(self.mean, dtype=cp.float32)

    def train_model(self, fold):
        train = cudf.read_parquet(
            self.train_paths,
            columns=self.features + [self.target]
        )

        X_train = (
            train[self.fold_df["fold"].to_numpy() != fold]
            [self.features].to_cupy().astype(cp.float32)
        )
        y_train = (
            train[self.fold_df["fold"].to_numpy() != fold]
            [self.target].to_cupy().astype(cp.float32)
        )

        X_valid = (
            train[self.fold_df["fold"].to_numpy() == fold]
            [self.features].to_cupy().astype(cp.float32)
        )

        X_train -= self.mean
        X_train /= (self.std + 1e-8)

        X_valid -= self.mean
        X_valid /= (self.std + 1e-8)

        model = Ridge(**self.params)
        model.fit(X_train, y_train)

        # === ここから: 係数とBiasの抽出 ===
        
        # 1. 係数 (Coefficients) の取得
        # GPU(CuPy)にある場合はCPU(NumPy)に戻す
        coefs = model.coef_
        if hasattr(coefs, "get"):
            coefs = coefs.get()
        # 形状が (1, n_features) のようになっている場合があるので1次元にならす
        coefs = coefs.ravel()

        # 2. 切片 (Bias / Intercept) の取得
        intercept = model.intercept_
        if hasattr(intercept, "get"):
            intercept = intercept.get() # スカラー値または1要素の配列
        # float型に変換しておく
        intercept = float(intercept)

        # 3. DataFrame作成 (XGBoostと同じ形式)
        # 特徴量の係数
        fi_df = pl.DataFrame({
            "Feature": self.features,
            "Importance": coefs
        })

        # Biasも表に含める場合（講義用ならあると分かりやすいです）
        bias_df = pl.DataFrame({
            "Feature": ["Intercept (Bias)"],
            "Importance": [intercept]
        }).with_columns(pl.col("Importance").cast(pl.Float32))
        
        # 縦に結合
        fi_df = pl.concat([fi_df, bias_df])

        return TrainResult(
            model=model,
            val_pred=model.predict(X_valid).get(),
            evals_result=None,
            extra=None,
            fi=fi_df,
            best_iteration=None
        )

    def predict_test(self, model):
        test = cudf.read_parquet(
            self.test_paths, columns=self.features
        ).to_cupy()

        test -= self.mean
        test /= (self.std + 1e-8)
        return model.predict(test).get()

    def train_on_all_data(self):
        pass
