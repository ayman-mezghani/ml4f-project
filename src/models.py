import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import tensorflow_probability as tfp
from tcn import TCN
from tensorflow import keras
from tensorflow.keras import layers


# Correlations for predicted and real
def MaxCorrelation(y_true, y_pred):
    """Goal is to maximize correlation between y_pred, y_true. Same as minimizing the negative."""
    mask = tf.math.not_equal(y_true, 0.)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    return -tf.math.abs(tfp.stats.correlation(y_true_masked, y_pred_masked, sample_axis=None, event_axis=None))


def Correlation(y_true, y_pred):
    return tf.math.abs(tfp.stats.correlation(y_pred, y_true, sample_axis=None, event_axis=None))


# Masked losses
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


def plot_training_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    histories = pd.DataFrame(history.history)
    epochs = list(range(1, len(histories) + 1))
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
    return fig


"""def prediction_details(predictions, y_test, asset_details, model_name,assets=range(Constants.N_ASSETS)):
    print('Asset:    Corr. coef.')
    print('---------------------')
    perf_df = pd.DataFrame(columns=['Model', 'asset', 'corr',
                                    'weights'])  # A FUNCTION THAT LOGS MODEL NAME, RMSE AND RMPSE INTO perf_df
    for i in assets:
        # drop first 14 values in the y_test, since they are absent in val_generator labels
        y_true = np.squeeze(y_test[Constants.WINDOW_SIZE - 1:, i])
        y_pred = np.squeeze(predictions[:, i])
        real_target_ind = np.argwhere(y_true != 0)

        #??
        assets_order = pd.read_csv('g-research-crypto-forecasting/supplemental_train.csv').Asset_ID[:14]
        assets_order = dict((t, i) for i, t in enumerate(assets_order))
        asset_id = list(assets_order.keys())[i]

        asset_name = asset_details[asset_details.Asset_ID == asset_id]['Asset_Name'].item()
        print(f"{asset_name}: {np.corrcoef(y_pred[real_target_ind].flatten(), y_true[real_target_ind].flatten())[0, 1]:.4f}")
        correl = np.corrcoef(y_pred[real_target_ind].flatten(), y_true[real_target_ind].flatten())[0, 1]
        weights = asset_details[asset_details.Asset_ID == i]['Weight'].item()

        perf_df = perf_df.append([pd.Series([model_name, asset_name, correl, weights], index=perf_df.columns)],
                                 ignore_index=True)

        plt.plot(y_true, label='Target')
        plt.plot(y_pred, label='Prediction')
        plt.xlabel('Time')
        plt.ylabel('Target')
        plt.legend()
        plt.show()
    return perf_df"""


def prediction_details(predictions, y_test, window_size, model_name, asset_details, assets):
    pred_details = []

    print('Asset:    Corr. coef.')
    print('---------------------')

    for i, asset in enumerate(assets):
        # drop first 14 values in the y_test, since they are absent in val_generator labels
        y_true = np.squeeze(y_test[window_size - 1:, i])
        y_pred = np.squeeze(predictions[:, i])

        real_target_ind = np.argwhere(y_true != 0)

        asset_name = asset_details[asset_details.Asset_ID == asset]['Asset_Name'].item()
        asset_weight = asset_details[asset_details.Asset_ID == i]['Weight'].item()
        corr = np.corrcoef(y_pred[real_target_ind].flatten(), y_true[real_target_ind].flatten())[0, 1]

        pred_details.append([model_name, asset_name, corr, asset_weight])

        print(f"{asset_name}: {corr:.4f}")
        plt.plot(y_true, label='Target')
        plt.plot(y_pred, label='Prediction')
        plt.xlabel('Time')
        plt.ylabel('Target')
        plt.title(asset_name)
        plt.legend()

    return pd.DataFrame(pred_details, columns=['Model_name', 'Asset_name', 'Correlation', 'Asset_weight'])


# Model 1
def get_model_LSTM(train_generator, n_assets):
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
    model = keras.Model(inputs=x_input, outputs=out, name='LSTM')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=masked_mse, metrics=[Correlation])
    return model


# Model 2
def get_model_Bidirectional_2_layer_LSTM(train_generator, n_assets):
    x_input = keras.Input(shape=(train_generator[0][0].shape[1], n_assets, train_generator[0][0].shape[-1]))
    branch_outputs = []

    for i in range(n_assets):
        a = layers.Lambda(lambda x: x[:, :, i])(x_input)  # Slicing the ith asset:
        a = layers.Masking(mask_value=0.)(a)
        a = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(a)
        a = layers.Bidirectional(layers.LSTM(16))(a)
        # a = layers.GlobalAvgPool1D()(a)
        branch_outputs.append(a)

    x = layers.Concatenate()(branch_outputs)
    x = layers.Dense(units=128)(x)
    out = layers.Dense(units=n_assets)(x)
    model = keras.Model(inputs=x_input, outputs=out, name='Bidirectional_2_layer_LSTM')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=masked_mse, metrics=[Correlation])
    return model


# 2lstm https://www.kaggle.com/ekaterinakryukova/g-research-parallel-lstm-training/edit
def get_model_Double_LSTM(train_generator, n_assets):
    x_input = keras.Input(shape=(train_generator[0][0].shape[1], n_assets, train_generator[0][0].shape[-1]))
    branch_outputs = []

    for i in range(n_assets):
        # Slicing the ith asset:
        a = layers.Lambda(lambda x_: x_[:, :, i])(x_input)
        a = layers.Masking(mask_value=0., )(a)
        a = layers.LSTM(units=32, return_sequences=True)(a)
        a = layers.LSTM(units=16)(a)
        branch_outputs.append(a)

    x = layers.Concatenate()(branch_outputs)
    x = layers.Dense(units=128)(x)
    out = layers.Dense(units=n_assets)(x)

    model = keras.Model(inputs=x_input, outputs=out, name='Double_LSTM')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=masked_cosine, metrics=[Correlation])

    return model


def get_model_LSTM_dropout(train_generator, n_assets):
    x_input = keras.Input(shape=(train_generator[0][0].shape[1], n_assets, train_generator[0][0].shape[-1]))
    branch_outputs = []

    for i in range(n_assets):
        # Slicing the ith asset:
        a = layers.Lambda(lambda x: x[:, :, i])(x_input)
        a = layers.Masking(mask_value=0., )(a)
        a = layers.LSTM(units=50, return_sequences=True)(a)
        a = layers.Dropout(0.2)(a)

        branch_outputs.append(a)

    x = layers.Concatenate()(branch_outputs)
    x = layers.Dense(units=128)(x)
    out = layers.Dense(units=n_assets)(x)
    # a = layers.LeakyReLU()(a)
    model = keras.Model(inputs=x_input, outputs=out, name='LSTM_dropout')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=masked_cosine, metrics=[Correlation])

    return model


def get_model_Conv1D_Double_LSTM(train_generator, n_assets):
    x_input = keras.Input(shape=(train_generator[0][0].shape[1], n_assets, train_generator[0][0].shape[-1]))
    branch_outputs = []

    for i in range(n_assets):
        # Slicing the ith asset:
        a = layers.Lambda(lambda x: x[:, :, i])(x_input)
        a = layers.Masking(mask_value=0., )(a)
        y = layers.Conv1D(32, 3, activation='relu', input_shape=input_shape[1:])(x)
        a = layers.LSTM(units=32, return_sequences=True)(a)
        a = layers.LSTM(units=16)(a)
        branch_outputs.append(a)

    x = layers.Concatenate()(branch_outputs)
    x = layers.Dense(units=128)(x)
    out = layers.Dense(units=n_assets)(x)

    model = keras.Model(inputs=x_input, outputs=out, name='Conv1D_Double_LSTM')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=masked_cosine, metrics=[Correlation])

    return model


def get_model_Triple_LSTM(train_generator, n_assets):
    x_input = keras.Input(shape=(train_generator[0][0].shape[1], n_assets, train_generator[0][0].shape[-1]))
    branch_outputs = []

    for i in range(n_assets):
        # Slicing the ith asset:
        a = layers.Lambda(lambda x: x[:, :, i])(x_input)
        a = layers.Masking(mask_value=0., )(a)
        a = layers.LSTM(units=32, return_sequences=True, dropout=0.2)(a)
        a = layers.LSTM(units=32, return_sequences=True, dropout=0.2)(a)
        a = layers.LSTM(units=32, dropout=0.2)(a)

        branch_outputs.append(a)

    x = layers.Concatenate()(branch_outputs)
    x = layers.Dense(units=128)(x)
    out = layers.Dense(units=n_assets)(x)

    model = keras.Model(inputs=x_input, outputs=out, name='Triple_LSTM')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=masked_cosine, metrics=[Correlation])

    return model


def get_model_TCN(train_generator, n_assets, dilatation=None, nb_filters=16):
    if dilatation is None:
        dilatation = [2, 4, 8, 16]

    x_input = keras.Input(shape=(train_generator[0][0].shape[1], n_assets, train_generator[0][0].shape[-1]))
    branch_outputs = []

    for i in range(n_assets):
        a = layers.Lambda(lambda x: x[:, :, i])(x_input)  # Slicing the ith asset:
        a = layers.Masking(mask_value=0.)(a)
        a = TCN(nb_filters=nb_filters, return_sequences=True, dilations=dilatation)(a)
        a = layers.GlobalAvgPool1D()(a)
        branch_outputs.append(a)

    x = layers.Concatenate()(branch_outputs)
    x = layers.Dense(units=128)(x)
    out = layers.Dense(units=n_assets)(x)
    model = keras.Model(inputs=x_input, outputs=out, name='TCN')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=masked_mse, metrics=[Correlation])
    return model


# Samples with a duration of WINDOW_SIZE records (minutes) will be formed from the train array.
# Each sample has a target vector corresponding to the final index if WINDOW_SIZE record.
class sample_generator(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, length):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.length = length
        self.size = len(x_set)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        for i in range(self.batch_size):
            start_ind = self.batch_size * idx + i
            end_ind = start_ind + self.length
            if end_ind <= self.size:
                batch_x.append(self.x[start_ind: end_ind])
                batch_y.append(self.y[end_ind - 1])
        return np.array(batch_x), np.array(batch_y)
