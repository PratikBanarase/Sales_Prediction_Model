# Sales Prediction Model

An end-to-end machine learning pipeline that predicts daily sales per store and product category from historical data. It covers data exploration, feature engineering, model training, hyperparameter tuning, evaluation and business interpretation.

## Project Structure

```
.
├── Sales_Prediction_Model.ipynb   # Main notebook (all 11 steps)
├── sales_data.csv                 # Input dataset
├── model_report.txt               # Performance summary and business interpretation
├── test_predictions.csv           # Actual vs predicted sales on the test set
├── best_sales_model.joblib        # Saved best model (created when the notebook runs)
├── figures/                       # Charts saved by the notebook
└── README_Sales_Prediction.md
```

## Dataset

`sales_data.csv` has 11,696 rows and 14 columns: daily sales for 4 stores and 4 product categories from 1 Jan 2023 to 31 Dec 2024.

| Column | Description |
|---|---|
| `date` | Date of sale |
| `store_id`, `city`, `region` | Store identifiers (`city` and `region` are determined by `store_id`) |
| `product_category` | Electronics, Clothing, Grocery, Home & Kitchen |
| `unit_price` | Average selling price |
| `discount_pct`, `promotion` | Discount percentage and promotion flag |
| `is_holiday` | Holiday flag |
| `marketing_spend` | Daily marketing spend |
| `avg_temperature_c` | Average temperature |
| `competitor_price` | Competitor's price |
| `units_sold` | Units sold (**excluded from features**, see below) |
| `sales` | **Target:** sales value |

The data has a small number of missing values in `marketing_spend`, `avg_temperature_c` and `competitor_price`, which are imputed in the pipeline.

> **Note:** If this file is the synthetic dataset supplied with the project, the scores below show how the pipeline behaves, not performance on real sales. The same notebook works on real data with the same columns.

## Pipeline Steps

| Step | What the notebook does |
|---|---|
| 1 | Load and explore the data: shape, dtypes, missing values, trends, distributions, correlations |
| 2 | Feature engineering: date components (year, month, day, day of week, week, quarter, weekend), lags (1, 7, 14 days), rolling means (7 and 30 days) |
| 3 | Missing values (median imputation) and categorical encoding (one-hot) inside a scikit-learn `ColumnTransformer` |
| 4 | Chronological 80/20 train/test split |
| 5 | Baseline: Linear Regression |
| 6 | Random Forest and XGBoost (optional) |
| 7 | Evaluation with RMSE, MAE and R² |
| 8 | Feature importance plots for the tree models |
| 9 | Actual vs predicted plots and residual analysis for the best model |
| 10 | Hyperparameter tuning with `RandomizedSearchCV` and `TimeSeriesSplit` |
| 11 | Performance report and business interpretation |

## Design Decisions

- **Chronological split:** The first 80% of dates train the model and the last 20% test it (train: 8,960 rows up to 12 Aug 2024; test: 2,256 rows from 13 Aug to 31 Dec 2024). A random split would leak future information through neighbouring days.
- **No target leakage:** `units_sold` is dropped because it is almost the same thing as `sales`. Rolling averages are shifted by one day so the current day is never used.
- **No preprocessing leakage:** Imputation, scaling and encoding are fitted on training data only.
- **Time-aware tuning:** Validation folds always come after their training folds.
- **Warm-up rows:** The first 14 to 30 days of each store-category series have no history and are dropped.

## Results (Test Set)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression (baseline) | 85,718 | 34,847 | 0.774 |
| Random Forest | 45,939 | 23,951 | 0.935 |
| XGBoost | 45,723 | 23,719 | 0.936 |
| Random Forest (tuned) | 46,110 | 23,909 | 0.935 |
| **XGBoost (tuned)** | **43,693** | **23,194** | **0.941** |

Best model: tuned XGBoost (`subsample=0.7`, `n_estimators=200`, `max_depth=3`, `learning_rate=0.03`, `colsample_bytree=0.85`).

- RMSE is 49% lower than the baseline and MAE is 33% lower.
- Average percentage error is about 16% per store, category and day, and about 7% on total daily sales.

## Key Insights

- **Recent sales history drives predictions.** Lag and rolling-average features account for about 66% of importance, and product, price and store for about 28%.
- **Promotions, holidays, marketing and weather show low direct importance** (about 3% combined). Their effect is already reflected in recent sales, so the lag features take the credit. To measure promotion or marketing return on investment, use a separate model without lag features.
- **Large orders are underpredicted.** The biggest Electronics sales fall below the diagonal in the actual vs predicted plot.
- **The model suits aggregate planning** (stock, staffing, budgets) better than exact single-item forecasts.

## Requirements

- Python 3.8+
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`
- `xgboost` (optional; the notebook runs without it)

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib xgboost jupyter
```

## How to Run

1. Place `sales_data.csv` in the same folder as the notebook.
2. Start Jupyter:
   ```bash
   jupyter notebook Sales_Prediction_Model.ipynb
   ```
3. Run all cells from top to bottom. Tuning takes a few minutes depending on your machine.

## Outputs

- `figures/01_eda.png`, `02_correlation.png`: exploration
- `figures/03_model_comparison.png`, `07_final_comparison.png`: model metrics
- `figures/04_importance_rf.png`, `04_importance_xgb.png`: feature importance
- `figures/05_actual_vs_predicted.png`, `06_residuals.png`: prediction quality
- `model_report.txt`, `test_predictions.csv`, `best_sales_model.joblib`

## Limitations and Next Steps

- Because of the lag features, forecasts beyond one day must be made recursively and accumulate error. For long horizons, retrain without short lags.
- Sales are heavily right-skewed, so RMSE is dominated by Electronics. Try per-category models or a log-transformed target.
- Two years of data only partly captures yearly seasonality.
- Retrain regularly (for example monthly) and monitor MAE to catch drift.
