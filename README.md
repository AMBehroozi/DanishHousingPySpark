# Danish Housing Price Prediction with PySpark

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PySpark](https://img.shields.io/badge/PySpark-3.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

## 📓 Open in Colab

[![Open Main Notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMBehroozi/DanishHousingPySpark/blob/main/DKHousingPricesPrediction.ipynb)
[![Open Data Visualization in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMBehroozi/DanishHousingPySpark/blob/main/data_visualization.ipynb)
[![Open Data Download in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AMBehroozi/DanishHousingPySpark/blob/main/get_data.ipynb)

---

A machine learning project that predicts Danish housing prices per square meter (DKK/m²) using Apache Spark and Gradient Boosted Trees regression. The model leverages both traditional features and advanced encoding techniques to achieve accurate predictions on real estate data.

## 📊 Project Overview

This project implements a scalable housing price prediction system using PySpark's distributed computing capabilities. It processes Danish housing market data and builds a predictive model that can estimate property prices based on location, property characteristics, and macroeconomic indicators.

### Key Features

- **Distributed Processing**: Built with Apache Spark for handling large-scale housing datasets
- **Advanced Feature Engineering**: 
  - Target encoding for high-cardinality features (city, zip code)
  - String indexing for low-cardinality categorical features
  - Temporal feature extraction (year, month, quarter)
- **Robust Data Cleaning**: Comprehensive missing value handling with intelligent imputation strategies
- **Temporal Train/Test Split**: Time-based data splitting to prevent data leakage
- **Production-Ready Model**: Gradient Boosted Trees with optimized hyperparameters

## 🎯 Model Performance

- **Test MAPE**: ~8-10% on 2023-2024 data
- **Training Period**: Data up to 2020
- **Validation Period**: 2021-2022
- **Test Period**: 2023-2024
- **Model**: Gradient Boosted Trees (60 iterations, depth=8)

## 📁 Project Structure

```
DanishHousingPySpark/
├── DKHousingPricesPrediction.ipynb  # Main prediction pipeline
├── DKHousingPrices.parquet          # Housing dataset (42.9 MB)
├── data_visualization.ipynb         # Exploratory data analysis
├── get_data.ipynb                   # Data download utility
├── data.yml                         # Conda environment specification
└── README.md                        # This file
```

## 🛠️ Technologies Used

- **Apache Spark 4.0.1**: Distributed data processing
- **PySpark ML**: Machine learning pipeline and algorithms
- **Python 3.14**: Core programming language
- **PyArrow 22.0**: Efficient data serialization
- **Pandas 2.3.3**: Data manipulation and analysis
- **NumPy 2.3.4**: Numerical computing
- **Matplotlib**: Data visualization

## 📋 Prerequisites

- Conda or Miniconda
- At least 64GB RAM (recommended for full dataset processing)
- 12+ CPU cores (recommended)

## 🚀 Getting Started

### 1. Environment Setup

Create and activate the conda environment:

```bash
conda env create -f data.yml
conda activate data
```

### 2. Data Preparation

The project includes a pre-processed Parquet file (`DKHousingPrices.parquet`). If you need to download or update the data, use:

```bash
jupyter notebook get_data.ipynb
```

### 3. Run the Prediction Pipeline

Open and execute the main notebook:

```bash
jupyter notebook DKHousingPricesPrediction.ipynb
```

## 📊 Dataset Features

### Categorical Features (Low Cardinality)
- `house_type`: Type of property (e.g., apartment, house)
- `sales_type`: Type of sale transaction
- `area`: Geographic area
- `region`: Danish region

### Categorical Features (High Cardinality)
- `city`: City name (target encoded)
- `zip_code`: Postal code (target encoded)

### Numerical Features
- `sqm`: Property size in square meters
- `%_change_between_offer_and_purchase`: Price negotiation metric
- `nom_interest_rate%`: Nominal interest rate
- `dk_ann_infl_rate%`: Danish annual inflation rate
- `yield_on_mortgage_credit_bonds%`: Mortgage bond yield
- `year_build`: Year of construction
- `no_rooms`: Number of rooms
- `year`: Sale year
- `month`: Sale month

### Target Variable
- `sqm_price`: Price per square meter (DKK/m²)

## 🔬 Machine Learning Pipeline

### 1. Data Loading & Preprocessing
- Load Parquet data with legacy timestamp handling
- Convert nanosecond epoch timestamps to proper dates
- Extract temporal features (year, month, quarter)
- Repartition and cache for optimal performance

### 2. Missing Value Handling
- Analyze missing value patterns across all features
- Drop rows with all features missing
- Remove rows with missing target values
- Impute numeric features with mean values
- Fill categorical features with "missing" label

### 3. Feature Encoding

**Target Encoding** (for high-cardinality features):
- Fit on training data only to prevent leakage
- Apply smoothing (parameter=10) to prevent overfitting
- Handle unseen categories with global mean
- Used for: `city`, `zip_code`

**String Indexing** (for low-cardinality features):
- Frequency-based ordering
- Handle invalid categories with "keep" strategy
- Used for: `house_type`, `sales_type`, `area`, `region`

### 4. Model Training
- Algorithm: Gradient Boosted Trees Regressor
- Target: log1p(sqm_price) for better distribution
- Hyperparameters:
  - Max iterations: 60
  - Max depth: 8
  - Max bins: 1024 (handles high cardinality)
  - Subsampling rate: 0.8
  - Feature subset strategy: sqrt
  - Loss type: squared

### 5. Evaluation
- Back-transform predictions from log scale
- Compute MAPE (Mean Absolute Percentage Error)
- Calculate RMSE, MAE, R² metrics
- Analyze feature importances
- Regional performance breakdown

## 📈 Key Insights

### Most Important Features
The model identifies the following as the most predictive features:
1. Geographic location (city, zip code via target encoding)
2. Property size (sqm)
3. Temporal features (year, month)
4. Macroeconomic indicators (interest rates, inflation)
5. Property characteristics (year built, number of rooms)

### Target Encoding Benefits
- Captures location-price relationships directly
- Reduces dimensionality compared to one-hot encoding
- Handles unseen categories gracefully
- Improves model performance on high-cardinality features

## 🔍 Data Exploration

For detailed exploratory data analysis, see:
```bash
jupyter notebook data_visualization.ipynb
```

This notebook includes:
- Distribution analysis of housing prices
- Geographic price variations
- Temporal trends
- Feature correlations
- Outlier detection

## ⚙️ Spark Configuration

The project uses optimized Spark settings for local execution:

```python
SPARK_CORES = 12              # Adjust based on your system
DRIVER_MEM = "64g"            # Adjust based on available RAM
SHUFFLE_PARTS = 320           # Optimized for dataset size
```

Additional optimizations:
- Adaptive query execution enabled
- Adaptive partition coalescing
- Kryo serialization for better performance
- Increased max result size (4GB)

## 📝 Usage Example

```python
# After training the model
# Make predictions on new data
predictions = gbt_model.transform(test_ready)

# Back-transform from log scale
predictions = predictions.withColumn(
    "pred_price_m2", 
    F.expm1(F.col("prediction"))
)

# View sample predictions
predictions.select(
    "city", "house_type", "sqm",
    "actual_price_m2", "pred_price_m2"
).show(10)
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional feature engineering
- Hyperparameter tuning
- Alternative models (Random Forest, XGBoost)
- Real-time prediction API
- Model deployment pipeline
- Extended temporal analysis

## 📄 License

This project is available for educational and research purposes.

## 🙏 Acknowledgments

- Danish housing market data sources
- Apache Spark community
- PySpark ML library contributors

## 📧 Contact

For questions or collaboration opportunities, please open an issue in the repository.

---

**Note**: This project demonstrates scalable machine learning techniques for real estate price prediction. The model should be regularly retrained with updated data to maintain accuracy.
