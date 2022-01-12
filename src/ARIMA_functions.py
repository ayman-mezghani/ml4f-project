import pandas as pd
from statsmodels.tsa.stattools import adfuller


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from itertools import product
import warnings
warnings.filterwarnings('ignore')

#!pip install pmdarima
import pmdarima as pm
from pmdarima.model_selection import train_test_split



from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import BoxCoxEndogTransformer
import pickle



def adf_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)

def plt_arima_result(y_train, forecasts ,y_test ,conf_int  ):
    """
    Plot predicted and real
    """
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)

    n_train = y_train.shape[0]
    x = np.arange(n_train + forecasts.shape[0])

    ax.plot(x[:n_train], y_train, color='blue', label='Training Data')
    ax.plot(x[n_train:], forecasts, color='green', marker='o',
            label='Predicted')
    ax.plot(x[n_train:], y_test, color='red', label='Actual')
    ax.legend(loc='lower left', borderaxespad=0.5)
    ax.set_title('Predicted Foo')
    ax.set_ylabel('# Foo')
    plt.fill_between(x[n_train:],
                 conf_int[:, 0], conf_int[:, 1],
                 alpha=0.1, color='b')

    plt.show()
#https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.pipeline.Pipeline.html
#https://github.com/alkaline-ml/pmdarima
def get_model_plot_pred(y,model_name,m,PCT_VALIDATION=0.8):
    # Load/split the data
    train_size=int(len(y)*PCT_VALIDATION)
    train, test = train_test_split(y, train_size=train_size)

    # Define and fit your pipeline
    pipeline = Pipeline([
        ('boxcox', BoxCoxEndogTransformer(lmbda2=1e-6)),  # lmbda2 avoids negative values
        ('arima', pm.AutoARIMA(seasonal=True, m =m, #min data
                               suppress_warnings=True,
                               trace=True))
    ])

    pipeline.fit(train)

    # Serialize your model just like you would in scikit:
    with open(model_name, 'wb') as pkl:
        pickle.dump(pipeline, pkl)

    # Load it and make predictions seamlessly:
    with open(model_name, 'rb') as pkl:
        mod = pickle.load(pkl)

    forecasts ,conf_int = mod.predict(len(y)-train_size,return_conf_int=True)
    plt_arima_result(y_train = train, forecasts = forecasts,y_test = test,conf_int=conf_int)
    cor = np.corrcoef(test, forecasts)[0,1]
    print(cor)
    return cor


