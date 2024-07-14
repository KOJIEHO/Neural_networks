import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Dropout
from keras.datasets import cifar10
from io import BytesIO
from PIL import Image
from keras.models import Model


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# __________________________________________________________________________________
'''
Пример построения простейшей нейронной сети с помощью пакета Keras.
Перевод градусов Цельсия в градусы Фаренгейта.
'''
# Задаем два множества для обучения
c = np.array([-40, -10, 0, 8, 15, 22, 38])
f = np.array([-40, 14, 32, 46, 59, 72, 100])

# Определяем модель НС, как последовательную (слои идут друг за другом)
model = keras.Sequential()
# В НС добавляем слой нейронов, состоящий из одного выходного нейрона (units=1), имеющего
# ровно один вход (input_shape=(1,)) и линейную активационную функцию (activation='linear'))
# Важно отметить, что bias автоматически добавляется для каждого нейрона
model.add(Dense(units=1, input_shape=(1,), activation='linear'))
# Компилируем структуру модели, указывая критерий качества (mean_squared_error) и способ оптимизации алгоритма
# градиентного спуска (Adam). Сеть автоматически инициализируется начальными значениями весов
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1))
# Запускаем обучение, используем метод .fit
log = model.fit(c, f, epochs=1000, verbose=False)

# Отображаем значения критерия качества (функция потерь - loss) в виде графика
plt.plot(log.history['loss'])
plt.grid(True)
plt.show()

# Подаем на вход произвольное значение
print(model.predict([100]))
# Выведем весовые коэффициенты
print(model.get_weights())
# __________________________________________________________________________________
'''
Пример нейронной сети для распознавания рукописные цифры.
Каждое изображение имеет размер 28*28 пикселей в градациях серого.
'''
# ========================================
# ======= Задача классификации (1) =======
# ===== Полносвязная нейронная сеть ======
# ================ MNIST =================
# ========================================

# Загружаем данные - здесь 60000 изображений в обучающей выборке, 10000 в тестовой
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Отображение первых 25 изображений из обучающей выборки
plt.figure(figsize=(10,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)
plt.show()

# Сформируем структуру нейронной сети. Имеем 784 входа, 128 нейронов скрытого слоя,
# 10 нейронов выходного слоя. В качестве функции активации скрытого слоя используем Relu.
# У выходных нейронов будем использовать softmax, т.к. необходимо интерпретировать выходные значения в терминах вероятности 
# принадлежности к тому или иному классу цифр.
model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),  # Входной слой
    Dense(128, activation='relu'),     # Скрытый слой
    Dense(10, activation='softmax')    # Выходной слой
]) 
print(model.summary())                 # Вывод структуры модели

# Нормализуем входные данные
x_train = x_train / 255
x_test = x_test / 255

# Переведем "ответы" в правильный формат выходных значений:
# [[1. 0. 0. ... 0. 0. 0.]  - вид "ответа" для 0
#  [0. 1. 0. ... 0. 0. 0.]  - вид "ответа" для 1
#  ...
#  [0. 0. 0. ... 0. 1. 0.]  - вид "ответа" для 8
#  [0. 0. 0. ... 0. 0. 9.]] - вид "ответа" для 9
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Выберем функцию потерь и способ оптимизации
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

# На данном этапе модель полностью подготовлена к обучению.
# Запустим процесс этого обучения.
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_split=0.2)
# Метод .evaluate прогоняет тестовое множество и вычисляет значение критерия качества и метрики
model.evaluate(x_test, y_test)

# Выполним разпознавание какого-либо тестового изображения 
n = 1                                      # Номер изображения (соответствует цифре "2")   
x = np.expand_dims(x_test[n], axis=0)      # Выбираем изображение из выборки
res = model.predict(x)                     # Прогоняем его через нейронную сеть
print(res)                                 # Получаем 10 возможнывх вариантов ответа
print(np.argmax(res))                      # Максимальное значение из этих 10 - нужное нам значение        
plt.imshow(x_test[n], cmap=plt.cm.binary)  # Изобразим на экране это тестовое изображение
plt.show()

# Пропустим через НС всю тестовую выборку, выделим неверные результаты распознавания
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)
# Сформируем маску, которая будет содержать True для верных вариантов,False - для неверных.
# С помощью этой маски выделим из тестовой выборки все неверные результаты
mask = pred == y_test
x_false = x_test[~mask]
y_false = x_test[~mask]

# Вывод первых 25 неверных результатов - те, где нейронка ошибочно определила значение цифры
plt.figure(figsize=(10,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_false[i], cmap=plt.cm.binary)
plt.show()

# __________________________________________________________________________________
# ========================================
# ======= Задача классификации (2) =======
# ====== Сверточная нейронная сеть =======
# ================ MNIST =================
# ========================================

# Загружаем данные - здесь 60000 изображений в обучающей выборке, 10000 в тестовой
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Нормализуем входные данные
x_train = x_train / 255
x_test = x_test / 255
# Переведем "ответы" в правильный формат выходных значений
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)
# Добавим еще одно измерение (одну ось) для цветовой компоненты (одноканального изображения)
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

# Сформируем структуру (модель) нейронной сети
model = keras.Sequential([
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10,  activation='softmax')
])
# Определим параметры компиляции и обучения
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

# Запустим процесс обучения
his = model.fit(x_train, y_train_cat, batch_size=500, epochs=5, validation_split=0.2)
# Прогоним через модель тестовые данные, проверим качество нейронной сети
model.evaluate(x_test, y_test_cat)

# Выполним разпознавание какого-либо тестового изображения 
n = 1                                      # Номер изображения (соответствует цифре "2")   
x = np.expand_dims(x_test[n], axis=0)      # Выбираем изображение из выборки
res = model.predict(x)                     # Прогоняем его через нейронную сеть
print(res)                                 # Получаем 10 возможнывх вариантов ответа
print(np.argmax(res))                      # Максимальное значение из этих 10 - нужное нам значение        
plt.imshow(x_test[n], cmap=plt.cm.binary)  # Изобразим на экране это тестовое изображение
plt.show()

# __________________________________________________________________________________
#=====================================
#======= Стилизация изображений ======
# ==== Сверточная нейронная сеть =====
#=====================================

# Загружаем контентное и стилевое изображения
img = Image.open('Земля.jpg')
img_style = Image.open('Солнце.jpg')

# Отображение изображений
plt.subplot(1, 2, 1)
plt.imshow( img )
plt.subplot(1, 2, 2)
plt.imshow( img_style )
plt.show()

# Преобразуем изображения во входной формат сети VGG19
x_img = keras.applications.vgg19.preprocess_input(np.expand_dims(img, axis=0))
x_style = keras.applications.vgg19.preprocess_input(np.expand_dims(img_style, axis=0))

# Определим функцию, которая вернет изображение в исходный формат RGB
def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)

    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                            "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")
  
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Загрузим обученную сеть VGG19, но без полносвязной НС на ее конце
# include_top=False - отбрасываем полносвязную сеть, weights='imagenet - загрузить веса,
# обученные на базе 10млн изображений, trainable = False - запрещаем изменять веса НС
vgg = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

# Выделим из НС выходы слоев с именами
content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

style_outputs = [vgg.get_layer(name).output for name in style_layers]
content_outputs = [vgg.get_layer(name).output for name in content_layers]
# Вычислим количество выходов слоев
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# Сформируем общий список выходных слоев
model_outputs = style_outputs + content_outputs

# Создадим новую сеть (копию) на базе VGG19 с требуемыми выходами
model = Model(vgg.input, model_outputs)
for layer in model.layers:
    layer.trainable = False
# print(model.summary())  # Вывод в консоль информацию о слоях

# Функция для выделения необходимых признаков для контентного и стилевого изображений
def get_feature_representations(model):
    # Пакетное вычисление содержимого и особенностей стиля
    style_outputs = model(x_style)
    content_outputs = model(x_img)
    # Получите представление о стиле и содержательной части из нашей модели
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    
    return style_features, content_features

# Функция для вычисления потерь формируемого изображения
def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

# Функция вычисления матрицы Грама для переданного ей тензора
def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)

    return gram / tf.cast(n, tf.float32)

# Функция вычисления квадратов рассогласований между картами стилей формируемого изображения и стилевого
def get_style_loss(base_style, gram_target):
    gram_style = gram_matrix(base_style)

    return tf.reduce_mean(tf.square(gram_style - gram_target))

# Общая функция вычисления всех потерь
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    model_outputs = model(init_image)
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]
    style_score = 0
    content_score = 0

    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)

    style_score *= style_weight
    content_score *= content_weight
    loss = style_score + content_score

    return loss, style_score, content_score

# Определим число итераций работы алгоритма, параметры для учета веса контента в формируемом изображении
num_iterations=30
content_weight=1e3
style_weight=1e-2
# Вычисляем карты стилей и контента для начальных изображений
style_features, content_features = get_feature_representations(model)
gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
# Определим матрицу Грама для начального стилевого изображения
init_image = np.copy(x_img)
init_image = tf.Variable(init_image, dtype=tf.float32)
# Указываем оптимизатор Adam, номер текущей итерации, переменные для хранения минимальных потерь
# и лучшего стилизованного изображения, кортеж параметров альфа и бета
opt = tf.compat.v1.train.AdamOptimizer(learning_rate=2, beta1=0.99, epsilon=1e-1)
iter_count = 1
best_loss, best_img = float('inf'), None
loss_weights = (style_weight, content_weight)
# Сформируем словарь конфигурации
cfg = {
      'model': model,
      'loss_weights': loss_weights,
      'init_image': init_image,
      'gram_style_features': gram_style_features,
      'content_features': content_features
}
# Всмопогательные переменные для преобразования
norm_means = np.array([103.939, 116.779, 123.68])
min_vals = -norm_means
max_vals = 255 - norm_means
imgs = []

# Запуск алгоритма градиентного спуска - формирование стилизованного изображения
for i in range(num_iterations):
    with tf.GradientTape() as tape:
       all_loss = compute_loss(**cfg)

    loss, style_score, content_score = all_loss
    grads = tape.gradient(loss, init_image)

    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)

    if loss < best_loss:
      # Update best loss and best image from total loss.
      best_loss = loss
      best_img = deprocess_img(init_image.numpy())

      # Use the .numpy() method to get the concrete numpy array
      plot_img = deprocess_img(init_image.numpy())
      imgs.append(plot_img)
      print('Iteration: {}'.format(i))

# Отображение стилизованного изображения
plt.imshow(best_img)
print(best_loss)

image = Image.fromarray(best_img.astype('uint8'), 'RGB')
image.save("result.jpg")
    