# Crop Predictor

# Crop Yield Predictor

This repository contains an end-to-end workflow for forecasting country-level crop yields using FAO data. It includes exploratory data analysis (EDA), model training, and an interactive Streamlit app for inference.

## Repository Structure

```
├── Jupyter notebook/                    # Jupyter notebook for EDA & supervised learning
│   └── crop_yield_predictor.ipynb
├── app.py                       # Streamlit application code
├── requirements.txt             # Python dependencies
├── crop_yield_cleaned.csv       # Cleaned FAO dataset used for charts & predictions
├── xgb_model.joblib             # Serialized XGBoost model pipeline
└── README.md                    # This file
```

### notebook/

* **crop\_yield\_analysis.ipynb**:

  * Data ingestion and cleaning steps
  * Exploratory Data Analysis (missingness, trends, feature distributions)
  * Supervised learning (model training, evaluation, comparison of Linear Regression vs. XGBoost)
  * Serialization of the final pipeline (`joblib.dump`)

### app.py

A Streamlit app that:

1. Loads the cleaned dataset and the trained XGBoost pipeline
2. Provides sidebar controls for selecting **Country**, **Crop**, **Historical Year Range**, and **Prediction Year**
3. Displays a historical yield trend chart
4. Runs the model to predict next-year yield and shows it as a metric
5. Optionally shows the raw historical data table

### requirements.txt

List of required Python packages:

```
streamlit
pandas
scikit-learn
xgboost
joblib
```

### crop\_yield\_cleaned.csv

The cleaned FAO crop yield dataset (preprocessed before train/test split). Used for both model training and interactive visualizations in `app.py`.

### xgb\_model.joblib

The serialized XGBoost pipeline (including preprocessing) saved via `joblib.dump`. Loaded by `app.py` to perform inference.

## Installation & Local Run

1. **Clone** this repo and `cd` into its root directory.
2. **Install** dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. **Run** the Streamlit app:

   ```bash
   streamlit run app.py
   ```

