
###Market
An average volume of $41 billion traded daily over the last year, according to CryptoCompare (as of 25th July 2021),  the market and meme manipulation, the correlation between assets and the very fast changing market conditions
###  Data Overview
Changes in prices between different cryptocurrencies are highly interconnected, extreme volatility of the assets, non-stationary
 From 2018-01-01 to 2021-09-21 for the majority of coins. For TRON, Stellar, Cardano, IOTA, Maker, and Dogecoin we have fewer data starting from later in 2018 or even later in 2019 in Dogecoin's case.
### Data features description

* timestamp: All timestamps are returned as second Unix timestamps (the number of seconds elapsed since 1970-01-01 00:00:00.000 UTC). Timestamps in this dataset are multiple of 60, indicating minute-by-minute data.
* Asset_ID
* Count: Total number of trades in the time interval (last minute).
* Open: Opening price of the time interval (in USD).
* High: Highest price reached during time interval (in USD).
* Low: Lowest price reached during time interval (in USD).
* Close: Closing price of the time interval (in USD).
* Volume: Quantity of asset bought or sold, displayed in base currency USD.
* VWAP: The average price of the asset over the time interval, weighted by volume. VWAP is an aggregated form of trade data.
* Target: Residual log-returns for the asset over a 15 minute horizon.
       Predict returns in the near future for prices $P^a$, for each asset $a$. $$R^a(t) = log (P^a(t+16)\ /\ P^a(t+1))$$
### EDA
* Missing asset data, for a given minute, is not represented by NaN's, but instead by the absence of those rows.
  there are many gaps in the data. To work with most time series models,the preprocess of data into a format without time gaps should be done. Missing data should be dropped
* Autocorrelation: The 'Close' prices seem to be mostly non-stationary. However, Bitcoin and Ethereum seem to be stationary. -> to make it stationary log_return - The log returns seem to be stationary
* For the majority of the coins it looks like we have a slight negative correlation at a lag of 1. For the other lags, it looks like the autocorrelations are statistically insignificant. This could indicate, that we have some Random-Walk behavior, which is going to make the prediction challenging.
* Correlation btc and eth:  high but variable correlation between the assets over time. correlation has proven difficult to maintain. For example, bitcoin prices fell even as prices for Ethereum’s ether (ETH) rose to new heights in early 2018.
* Trend: negative peak around March 2020
### TO DO
* Plot with diff moving average
* Add more data feat:
    * given time-series of historical prices  = millions of rows of minute-by-minute cryptocurrency trading data
    * decompose date (timestamp)
    * upper/ lower shadow
    * use quora, reddit, twitter, EDGAR Data for NLP
    * reduce the data memory ?
    * https://www.kaggle.com/yamqwe/time-series-modeling-n-beats
* Stat tests:
    * autocorrelation, time-series decomposition and stationarity tests.
* EDA: relationship between the different coins (close/ avg/ open per day) depending on the year,
* Train_test split with diff time frames / paddles ; diff windows
* Add more Preprocess
    * Standardization StandardScaler(), MinMax
    * Sliding windows over a timeseries https://keras.io/api/preprocessing/timeseries/
* Models:
    * Baseline model: Linear Regression, MultiOutputRegressor(LinearRegression()), Ridge()
    * LGBMRegressor
        * lgb_params = {
"objective": "regression",
"n_estimators" : 500,     # <-- (9) change from 200 to 500
"num_leaves" : 300,       # <-- (10) Added parameter
"learning_rate" : 0.09,   # <-- (10) Added parameter
"random_seed" : 1234}
        * change use of model from "for all data" to "for each asset"
    * N-BEATS https://arxiv.org/pdf/1905.10437.pdf https://www.kaggle.com/yamqwe/time-series-modeling-n-beats
    * SARIMA, AR, MA, ARMA, ARIMA and others https://towardsdatascience.com/time-series-models-d9266f8ac7b0
        * from darts.models import (
  NaiveSeasonal,
  NaiveDrift,
  Prophet,
  ExponentialSmoothing,
  ARIMA,
  AutoARIMA,
  RegressionEnsembleModel,
  RegressionModel,
  FFT)
    * XGBoost https://www.kaggle.com/yamqwe/crypto-prediction-xgb-regressor#Training-🏋%EF%B8%8F
    * Recurrent neural networks (RNN)
    * RNN variants (LSTM, GRU) https://keras.io/api/layers/recurrent_layers/
    * TCN https://github.com/philipperemy/keras-tcn
(week 6)
* Val by weighted correlation
* Tune hyperparams
* Submission: Note that this is a Code Competition,
in which you must submit your notebook to be run against the hidden private data.
Your notebook should use the provided python time-series API, which ensures that models do not peek forward in time. To use the API, follow the instructions and template in Code Competition Detailed API instructions and Basic Submission Template.

### Useful links
* Applying Deep Neural Networks to Financial Time Series Forecasting http://infosci.cornell.edu/~koenecke/files/Deep_Learning_for_Time_Series_Tutorial.pdf
* Tutorial to the G-Research Crypto Competition
https://www.kaggle.com/cstein06/tutorial-to-the-g-research-crypto-competition#Crypto-forecasting-tutorial
* G-Research Crypto Forecasting EDA
https://www.kaggle.com/iamleonie/to-the-moon-g-research-crypto-forecasting-eda#Data-Overview
* Time Series: Interpreting ACF and PACF https://www.kaggle.com/iamleonie/time-series-interpreting-acf-and-pacf
* Hyperparam tune https://www.kaggle.com/junjitakeshima/crypto-beginner-s-try-for-simple-lgbm-en-jp
* Complex models: https://www.kaggle.com/yamqwe/time-series-modeling-n-beats
* Datart https://www.kaggle.com/cecileguillot/naive-drift-and-regression-approaches
