import pandas as pd
import numpy as np
import gc
def c_time_sub(asset_id, data):
    df=data[data["Asset_ID"]==asset_id].set_index("timestamp")
    df=df.reindex(range(df.index[0],df.index[-1]+60,60), method="pad")
    df.index = pd.to_datetime(df.index, unit='s')
    return df
def vwap_change(train):
    VWAP_max = np.max(train[np.isfinite(train.VWAP)].VWAP)
    VWAP_min = np.min(train[np.isfinite(train.VWAP)].VWAP)
    train['VWAP'] = np.nan_to_num(train.VWAP, posinf=VWAP_max, neginf=VWAP_min)
    return train
def process_all_assets(train):
    train['timestamp'] = pd.to_datetime(train['timestamp'], unit='s')
    train.index = train.timestamp
    train.sort_index(inplace=True)
    train['is_real'] = True
    ind = train.index.unique()

    def reindex(df):
        res = df.reindex(pd.date_range(ind.min(), ind.max(), freq='min'))
        res['is_real'].fillna(False, inplace=True)
        res['timestamp'] = res.index
        res = res.fillna(method="ffill").fillna(method="bfill")
        return res

    train = train.groupby('Asset_ID').apply(reindex).reset_index(0, drop=True).sort_values(by=['timestamp', 'Asset_ID'])
    gc.collect()
    return train


def feature_eng(df, row=False):
    _df = df.copy()

    _df['Spread'] = _df['High'] - _df['Low']
    _df['Close-Open'] = _df['Close'] - _df['Open']

    _df['Upper_Shadow'] = _df['High'] - np.maximum(_df['Close'], _df['Open'])
    _df['Lower_Shadow'] = np.minimum(_df['Close'], _df['Open']) - _df['Low']

    feature_cols = _df.columns.drop(['Asset_ID', 'timestamp', 'is_real'])
    for col in feature_cols:
        _df[col] = _df[col] * _df.is_real
    _df = _df[feature_cols]


    return _df

