import numpy as np
import pandas as pd
import collections
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Input
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.preprocessing import timeseries_dataset_from_array


#=====================================
#============ SPAM / HAM =============
#=====================================
# Загрузка датасета и разделение датасета на тренировочную и тестовую выборки
dataset = pd.read_csv('SPAM_text_message_20170820_-_Data.csv')
dataset['Category'] = dataset['Category'].map({'spam': 1, 'ham': 0})
data_X, data_y = dataset['Message'], dataset['Category']
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.1, random_state=42)

# Преобразование текстовых данных в последовательности индексов и их паддинг
tokenizer = Tokenizer(num_words=4000, lower=True, split=' ')
tokenizer.fit_on_texts(X_train)

X_train_sequence = tokenizer.texts_to_sequences(X_train)
X_train_pad = pad_sequences(X_train_sequence, maxlen=100)

X_test_sequence = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(X_test_sequence, maxlen=100)

# Создание модели LSTM
network = Sequential([
    Embedding(4000, 128, input_length=100),
    SpatialDropout1D(0.2),
    LSTM(50, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])
network.compile(optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])
network.summary()

# Обучение модели
early_stop = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)
history = network.fit(X_train_pad, y_train, epochs=10, batch_size=64,
                      validation_split=0.1, callbacks=[early_stop])

# Оценка эффективности модели
loss, accuracy = network.evaluate(X_test_pad, y_test)
print(loss, accuracy)

# Проверяем работу
new_text = ["Your free ringtone is waiting to be collected. Sorry, I'll call later in meeting."]
new_seq = tokenizer.texts_to_sequences(new_text)
new_X = pad_sequences(new_seq, maxlen=100)

prediction = network.predict(new_X)
print(prediction)

# __________________________________________________________________________________
#=====================================
#====== Jena_climate_2009_2016 =======
#=====================================
# Загрузка датасета и его нормализация
dataset = pd.read_csv('jena_climate_2009_2016.csv')
features_cols = ['p (mbar)', 'VPmax (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'rho (g/m**3)', 'wv (m/s)']
target_col = "T (degC)"
cols = features_cols + [target_col]
dataset[features_cols] = (dataset[features_cols] - dataset[features_cols].mean()) / dataset[features_cols].std()

# Создание временных рядов (тренировочный и валидационный временные ряды)
data_split = int(0.9 * int(dataset.shape[0]))
train_start = 720 + 72
train_end = train_start + data_split
sequence_len = int(720 / 6)

data_train = dataset.iloc[0:data_split - 1][cols]
X_train = data_train[cols].values
y_train = dataset.iloc[train_start:train_end][target_col]
train_timeseries = timeseries_dataset_from_array(
    data=X_train,
    targets=y_train,
    sequence_length=sequence_len,
    sampling_rate=6,
    batch_size=1024,
)

data_val = dataset.iloc[data_split:][cols]
test_start = len(data_val) - train_start
X_val = dataset.iloc[:test_start][cols].values
y_val = dataset.iloc[train_end:][target_col]
val_timeseries = timeseries_dataset_from_array(
    data=X_val,
    targets=y_val,
    sequence_length=sequence_len,
    sampling_rate=6,
    batch_size=1024,
)

# Создание модели LSTM
network = Sequential([
    Input(shape=(sequence_len, len(cols))),
    LSTM(128, return_sequences=True),
    LSTM(32, return_sequences=False),
    Dense(32, activation='swish'),
    Dense(1)
])
network.compile(optimizer=Adam(learning_rate=0.001),
                loss='mse', metrics=['mape', 'mae'])
network.summary()

# Обучение модели
history = network.fit(train_timeseries, epochs=5,
                      validation_data=val_timeseries)

# Оценка эффективности модели
loss = network.evaluate(val_timeseries)
print(loss, accuracy)