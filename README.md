# Kaggle Tabular Competition Templates 🏆

Kaggleのテーブルデータコンペティション用のベースラインモデルおよびトレーニングパイプラインのテンプレート集です。
GBDT, Neural Networks, そして GPU加速された cuML モデルを含んでいます。

## 📦 Supported Models

以下のモデルが実装されています。

### Gradient Boosting Decision Trees (GBDT)
- **`xgb`**: XGBoost
- **`lgbm`**: LightGBM
- **`cb`**: CatBoost

### Neural Networks (Deep Learning)
- **`mlp`**: Multi-Layer Perceptron
- **`realmlp`**: RealMLP (ResNet-like architecture for tabular)
- **`tabnet`**: TabNet

### Classical Machine Learning (GPU Accelerated via cuML)
以下のモデルは `cuml` を使用しており、**GPU環境が必須**です。
- **`logreg`**: Logistic Regression
- **`rfr`**: Random Forest Regressor
- **`rfc`**: Random Forest Classifier
- **`ridge`**: Ridge Regression
- **`lasso`**: Lasso Regression
- **`svc`**: Support Vector Classifier

---

## ⚠️ Important Usage Notes (必ずお読みください)

このテンプレートを使用する際は、以下の4点をタスクに合わせて必ず修正・確認してください。

### 1. Data Preparation (`fold` column)
入力データフレームには、CV（Cross Validation）用の **`fold` 列が必須**です。
事前に StratifiedKFold や GroupKFold などで `fold` を割り振ってからデータを渡してください。


### 2. Adjust Objectives (Params)
各モデルのTrainer内にある params の objective (損失関数) は、タスク（二値分類、多クラス分類、回帰など）に合わせて変更してください。

Binary Classification: binary:logistic, Logloss, etc.

Regression: reg:squarederror, RMSE, etc.

3. Adjust Metrics (Base CV Trainer)
BaseCVTrainer 内で定義されている評価指標（Metric）もタスクに応じて変更する必要があります。


# BaseCVTrainer or config
self.metric = ... # e.g., mean_squared_error, roc_auc_score
4. Hardware Requirement (GPU)
cuml ベースのモデル（SVC, Ridge, Lasso, RFなど）および Deep Learning モデルは GPU環境 での実行を前提として設定されています。CPU環境では動作しない、または設定の変更が必要です。

🛠 Installation
必要なライブラリのバージョンは environment.yaml に記載されています。 conda 環境を作成して使用してください。

conda env create -f environment.yaml
conda activate <env_name>

# 🚀 Workflow
このテンプレートは以下の順序で実行することを想定しています。

## 1. Feature Engineering
notebooks/fe/ 以下のNotebookで特徴量を作成します。

Output: artifacts/features/{data_id}/train.parquet および meta.json

Rule: 作成するデータには必ずCV用の fold 列を含めてください。

## 2. Hyperparameter Tuning (Optuna)
モデルと作成したデータIDを指定してOptunaを実行します。

Output: artifacts/optuna/{model}-{data_id}/trl{n}.json

Note: このJSONにはハイパーパラメータだけでなく、探索時のメタ情報も含まれます。学習時にはここから必要なパラメータをロードします。

## 3. Training & CV
notebooks/training/02_gpu_cv.ipynb (または対応するスクリプト) を使用して学習を行います。 Optunaで特定した trl{n}.json を指定して、OOFおよびTest予測を作成します。

Input: Feature (data_id), Params (trl{n}.json)

Output: runs/{model}-{data_id}-trl{n}-{fold}fold-s{seed}/

このディレクトリに oof_pred, test_pred, 特徴量重要度のプロット、Notebookのスクリーンショット等が保存されます。

## 4. Submission / Ensemble
notebooks/others/submission.ipynb にて、runs/ ディレクトリ内の test_pred (または oof_pred を使ったEnsemble結果) を読み込み、提出用ファイルを作成します。