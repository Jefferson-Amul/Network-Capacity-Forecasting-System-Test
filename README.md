

\# Network Capacity Forecasting System



\## Overview

A machine learning project that forecasts future network traffic volume using historical network traffic data. It supports network capacity planning by helping identify potential increases in traffic demand.



\## Features

\- Network data loading and quality checks

\- Feature engineering and preprocessing

\- TensorFlow-based traffic forecasting

\- Chronological training, validation, and testing

\- Model evaluation using MAE, RMSE, and R²

\- Feature importance analysis

\- MLflow experiment tracking

\- Streamlit dashboard with interactive charts



\## Technologies

Python, Pandas, NumPy, Scikit-learn, TensorFlow, MLflow, Streamlit, and Plotly.



\## Setup

1\. Install Python 3.12.

2\. Create and activate a virtual environment.

3\. Install dependencies with `pip install -r requirements.txt`.

4\. Obtain the CESNET-TimeSeries24 dataset and place the required CSV at `data/institutions/agg\_10\_minutes/0.csv`.

5\. Run `python main.py` to train the model.

6\. Run `python -m streamlit run dashboard.py` to launch the dashboard.



\## Dataset

The dataset is not included in this repository. Obtain it from its authorized source and follow its usage terms.



\## Limitations

This is an academic prototype. Forecast quality depends on the input data and evaluation conditions. Predictions are intended to support capacity planning, not replace operational monitoring.



