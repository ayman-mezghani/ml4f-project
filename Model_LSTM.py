import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
#import gresearch_crypto
import gc

import Constants

#Samples with a duration of WINDOW_SIZE records (minutes) will be formed from the train array. Each sample has a target vector corresponding to the final index if WINDOW_SIZE record.
class sample_generator(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, length):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.length = length
        self.size = len(x_set)
    def __len__(self): return int(np.ceil(len(self.x) / float(self.batch_size)))
    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        for i in range(self.batch_size):
            start_ind = self.batch_size * idx + i
            end_ind = start_ind + self.length
            if end_ind <= self.size:
                batch_x.append(self.x[start_ind : end_ind])
                batch_y.append(self.y[end_ind -1])
        return np.array(batch_x), np.array(batch_y)


#Correlations for predicted and real
def MaxCorrelation(y_true,y_pred):
    return -tf.math.abs(tfp.stats.correlation(y_pred, y_true, sample_axis=None, event_axis=None))

def Correlation(y_true,y_pred):
    return tf.math.abs(tfp.stats.correlation(y_pred, y_true, sample_axis=None, event_axis=None))

#Masked losses
def masked_mse(y_true, y_pred):
    mask = tf.math.not_equal(y_true, 0.)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    return tf.keras.losses.mean_squared_error(y_true=y_true_masked, y_pred=y_pred_masked)

def masked_mae(y_true, y_pred):
    mask = tf.math.not_equal(y_true, 0.)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    return tf.keras.losses.mean_absolute_error(y_true=y_true_masked, y_pred=y_pred_masked)

def masked_cosine(y_true, y_pred):
    mask = tf.math.not_equal(y_true, 0.)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    return tf.keras.losses.cosine_similarity(y_true_masked, y_pred_masked)


# Model
def get_modell(X_train, y_train,n_assets=Constants.N_ASSETS):
    train_generator = sample_generator(X_train, y_train, length=Constants.WINDOW_SIZE, batch_size = Constants.BATCH_SIZE)
    x_input = keras.Input(shape=(train_generator[0][0].shape[1], n_assets, train_generator[0][0].shape[-1]))
    branch_outputs = []

    for i in range(n_assets):
        a = layers.Lambda(lambda x: x[:, :, i])(x_input)  # Slicing the ith asset:
        a = layers.Masking(mask_value=0.)(a)
        a = layers.LSTM(units=32, return_sequences=True)(a)
        a = layers.GlobalAvgPool1D()(a)
        branch_outputs.append(a)

    x = layers.Concatenate()(branch_outputs)
    x = layers.Dense(units=128)(x)
    out = layers.Dense(units=n_assets)(x)
    model = keras.Model(inputs=x_input, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=masked_mse, metrics=[Correlation])
    return model


def plot_training_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    histories = pd.DataFrame(history.history)
    epochs = list(range(1, len(histories)+1))
    loss = histories['loss']
    val_loss = histories['val_loss']
    correlation = histories['Correlation']
    val_correlation = histories['val_Correlation']
    ax[0].plot(epochs, loss, label='Train Loss')
    ax[0].plot(epochs, val_loss, label='Val Loss')
    ax[0].set_title('Losses')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(loc='upper right')
    ax[1].plot(epochs, correlation, label='Train Correlation')
    ax[1].plot(epochs, val_correlation, label='Val Correlation')
    ax[1].set_title('Correlations')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(loc='upper right')
    fig.show()
    gc.collect()

def prediction_details(predictions, y_test, asset_details, assets=range(N_ASSETS =Constants.N_ASSETS)):
    print('Asset:    Corr. coef.')
    print('---------------------')
    for i in assets:
        # drop first 14 values in the y_test, since they are absent in val_generator labels
        y_true = np.squeeze(y_test[WINDOW_SIZE - 1:, i])
        y_pred = np.squeeze(predictions[:, i])
        real_target_ind = np.argwhere(y_true != 0)
        asset_name = asset_details[asset_details.Asset_ID == i]['Asset_Name'].item()
        print(f"{asset_name}: {np.corrcoef(y_pred[real_target_ind].flatten(), y_true[real_target_ind].flatten())[0, 1]:.4f}")
        plt.plot(y_true, label='Target')
        plt.plot(y_pred, label='Prediction')
        plt.xlabel('Time')
        plt.ylabel('Target')
        plt.legend()
        plt.show()