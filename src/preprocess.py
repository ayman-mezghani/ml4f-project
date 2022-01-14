import gc

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import pandas_ta as ta # to be used for more feature engineering


def c_time_sub(asset_id, data):
    df = data[data["Asset_ID"] == asset_id].set_index("timestamp")
    df = df.reindex(range(df.index[0], df.index[-1] + 60, 60), method="pad")
    df.index = pd.to_datetime(df.index, unit='s')
    return df


def add_features(_df):
    _df['Log_close_diff'] = np.log(_df['Close']).diff()
    _df['Log_open_diff'] = np.log(_df['Open']).diff()
    _df['Log_high_diff'] = np.log(_df['High']).diff()
    _df['Log_low_diff'] = np.log(_df['Low']).diff()
    _df['Log_vwap_diff'] = np.log(_df['VWAP']).diff()
    
    _df['Spread'] = _df['High'] - _df['Low']
    _df['Close_Open'] = _df['Close'] - _df['Open']

    _df['Upper_Shadow'] = _df['High'] - np.maximum(_df['Close'], _df['Open'])
    _df['Lower_Shadow'] = np.minimum(_df['Close'], _df['Open']) - _df['Low']

    _df['Close/Open'] = _df['Close'] / _df['Open']
    _df['log_Close/Open'] = np.log(_df['Close/Open'])

    _df['Mean_trade'] = _df['Volume'] / (_df['Count'] + 1)
    
    _df.ta.macd(close='close', append=True)

    _df.ta.coppock(close='close', append=True)

    _df.ta.stochrsi(close='close', append=True)

    _df.ta.dpo(close='close', append=True)
    
    _df['High/Low'] = _df['High'] / _df['Low']
    _df['LOG_High/Low'] = np.log(_df['High/Low'])

    _df['LOG_VOL'] = np.log(1. + _df['Volume'])
    _df['LOG_CNT'] = np.log(1. + _df['Count'])

    _df['Mean'] = _df[['Open', 'High', 'Low', 'Close']].mean(axis = 1)
    _df['High/Mean'] = _df['High'] / _df['Mean']
    _df['Low/Mean'] = _df['Low'] / _df['Mean']

    _df['Median'] = _df[['Open', 'High', 'Low', 'Close']].median(axis=1)
    _df['High/Median'] = _df['High'] / _df['Median']
    _df['Low/Median'] = _df['Low'] / _df['Median']

    return _df


def process_all_assets(_df, scaler=RobustScaler()):
    df = _df.copy()
    # casting to float
    df[['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Target']] = \
        df[['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Target']].astype(np.float32)

    # fill all the data with zero
    df['Target'] = df['Target'].fillna(0)

    # VWAP column has -inf and inf values. VWAP_max and VWAP_min will be used for replacement
    VWAP_max = np.max(df[np.isfinite(df.VWAP)].VWAP)
    VWAP_min = np.min(df[np.isfinite(df.VWAP)].VWAP)
    df['VWAP'] = np.nan_to_num(df.VWAP, posinf=VWAP_max, neginf=VWAP_min)

    df.index = df.timestamp
    df.sort_index(inplace=True)
    df['is_real'] = True

    # add feat
    df = add_features(df)

    feature_cols = df.columns.drop(['Asset_ID', 'Target', 'timestamp', 'is_real'])

    # scale
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # reindexing, frequency is minutes
    ind = df.index.unique()
    def reindex(df):
        res = df.reindex(pd.date_range(ind.min(), ind.max(), freq='min'))
        res['is_real'].fillna(False, inplace=True)
        res['timestamp'] = res.index
        res = res.fillna(method="ffill").fillna(method="bfill")
        return res

    df = df.groupby('Asset_ID').apply(reindex).reset_index(0, drop=True).sort_values(by=['timestamp', 'Asset_ID'])
    gc.collect()

    # Feature values for 'non-real' rows are set to zeros
    """for col in feature_cols:
        df[col] = df[col] * df.is_real"""

    return df
