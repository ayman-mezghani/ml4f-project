# G-Research Crypto Forecasting

Ekaterina Kryukova, Ayman Mezghani

### Competition Details
* [Link](https://www.kaggle.com/c/g-research-crypto-forecasting)
* [Data](https://www.kaggle.com/c/g-research-crypto-forecasting/data)

### How-To
- Data is to be placed under a directory called `data/` in the following way:<br/>
```
data
└───g-research-crypto-forecasting
    |   asset_details.csv
    |   example_sample_submission.csv
    |   example_test.csv
    |   supplemental_train.csv
    |   train.csv
```
- `notebooks/` contains the following notebooks:
  - `EDA.ipynb`: containing the exploratory data analysis
  - `Arima_auto.ipynb`: containing the ARIMA model
  - `ANN_methods.ipynb` : containing the ANN models for forecasting
- `src` contains the python scripts
- `environment.yml` is to be used when creating the environment to run the code
