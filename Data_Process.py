import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import RobustScaler

def c_time_sub(asset_id, data):
    df=data[data["Asset_ID"]==asset_id].set_index("timestamp")
    df=df.reindex(range(df.index[0],df.index[-1]+60,60), method="pad")
    df.index = pd.to_datetime(df.index, unit='s')
    return df
def add_features(df, row=False):
    _df = df.copy()

    _df['Spread'] = _df['High'] - _df['Low']
    _df['Close_Open'] = _df['Close'] - _df['Open']

    _df['Upper_Shadow'] = _df['High'] - np.maximum(_df['Close'], _df['Open'])
    _df['Lower_Shadow'] = np.minimum(_df['Close'], _df['Open']) - _df['Low']
    _df['spread'] = _df['High'] - _df['Low']
    _df['mean_trade'] = _df['Volume'] / _df['Count']
    _df['log_price_change'] = np.log(_df['Close'] / _df['Open'])


    return _df



def process_all_assets(train):
    # for assets sorting
    assets_order = pd.read_csv('g-research-crypto-forecasting/supplemental_train.csv').Asset_ID[:14]
    assets_order = dict((t, i) for i, t in enumerate(assets_order))

    train[['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Target']] = \
        train[['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Target']].astype(np.float32)
    train['Target'] = train['Target'].fillna(0)

    VWAP_max = np.max(train[np.isfinite(train.VWAP)].VWAP)
    VWAP_min = np.min(train[np.isfinite(train.VWAP)].VWAP)
    train['VWAP'] = np.nan_to_num(train.VWAP, posinf=VWAP_max, neginf=VWAP_min)



    # Get the series of the 'real' record ids for futher matching

    df = train[['Asset_ID', 'Target']].copy()
    times = dict((t, i) for i, t in enumerate(df.index.unique()))
    df['id'] = df.index.map(times)
    df['id'] = df['id'].astype(str) + '_' + df['Asset_ID'].astype(str)
    ids = df.id.copy()
    del df






    #add feat

    train = add_features(train)

    #scale
    scale_features = train.columns.drop(['Asset_ID', 'Target'])
    RS = RobustScaler()
    train[scale_features] = RS.fit_transform(train[scale_features])

    ind = train.index.unique()
    #fill gaps
    def reindex(df):
        df = df.reindex(range(ind[0], ind[-1] + 60, 60), method='nearest')
        df = df.fillna(method="ffill").fillna(method="bfill")
        return df

    train = train.groupby('Asset_ID').apply(reindex).reset_index(0, drop=True).sort_index()


    # Matching records and marking generated rows as 'non-real'

    train['group_num'] = train.index.map(times)
    train = train.dropna(subset=['group_num'])
    train['group_num'] = train['group_num'].astype('int')

    train['id'] = train['group_num'].astype(str) + '_' + train['Asset_ID'].astype(str)

    train['is_real'] = train.id.isin(ids) * 1
    train = train.drop('id', axis=1)
    #print(train)
    # Features values for 'non-real' rows are set to zeros

    features = train.columns.drop(['Asset_ID', 'group_num', 'is_real'])
    train.loc[train.is_real == 0, features] = 0.

    # Sorting assets according to their order in the 'supplemental_train.csv'

    train['asset_order'] = train.Asset_ID.map(assets_order)
    train = train.sort_values(by=['group_num', 'asset_order'])
    return train



