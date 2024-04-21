import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns


#-------------------------------------
#---------- Основы обучения ----------
#------------------------------------- 
#_______________________________________________________________________________________________
'''
Пример работы самого простого случая  НС прямого распространения с двумя нейронами.
Предположим, что некая девочка выбирает себе парня по трем параметрам:
1) Наличие квартиры 2) Отношение к тяжелому року 3) Красота
Да - 1, нет - 0
Предположим, что девочка положительно относится к наличию квартиры и красоте,
но негативно к року. Это повлияет на веса связей - для рока вес связи будет отрицательным, 
а для дома и красоты вес будет положительным.
На входе формируется суммарный сигнал из суммы входящих параметров помноженных на свои веса.
Далее это значение проходит через функцию активации,  после который формируется "Да" или "Нет".
'''
def act(x): # Функция возвращает 0 или 1 в зависимости от
    return 0 if x < 0.5 else 1
 
def go(house, rock, attr):
    x = np.array([house, rock, attr])
    w11 = [0.3, 0.3, 0]              # Весовые кф для первого нейрона
    w12 = [0.4, -0.5, 1]             # Весовые кф для второго нейрона
    weight1 = np.array([w11, w12])   # Матрица кф 2x3
    weight2 = np.array([-1, 1])      # Вектор кф 1х2
    
    # Вычисляем сумму на входах нейронов скрытого слоя
    sum_hidden = np.dot(weight1, x)  
    print("Значения сумм на нейронах скрытого слоя: "+str(sum_hidden))
    
    # Вычисляем значения на выходах из нейронов скрытого слоя
    out_hidden = np.array([act(x) for x in sum_hidden])
    print("Значения на выходах нейронов скрытого слоя: "+str(out_hidden))
    
    # Вычисляем сумму на последнем слое
    sum_end = np.dot(weight2, out_hidden)
    y = act(sum_end)
    print("Выходное значение НС: "+str(y))
 
    return y

# Задаем начальные значения
house = 1
rock = 1
attr = 0
 
res = go(house, rock, attr)  # Если сумма на последнем слое больше 0.5, то вернет 1
if res == 1:                 # Если вернется 1, значит мальчик с такими интересами подохдит девочке
    print("ОК")
else:
    print("NOT OK")
#_______________________________________________________________________________________________
'''
Пример работы простейшего персептрона для задачи классификации двух классов образов.
Присутствует один нейрон. Модель проводит одну разделяющую прямую между этими классами.
Эта линия будет проходит через точку начала координат. Предположим, что оба класса сдвигаются 
по какой-то оси. Тогда прямая не сможет разделить классы, т.к. она проходит через начало координат.
По этой причине во всех НС вводится понятие смещения - дополнительно определяют еще
один вход для смещения разделяющей гиперплоскости (bias). 
'''
N = 5  # Кол-во элементов внутри одного класса
b = 3  # Смещение
 
# Генерируем первый и второй классы для последуюущего разделения
x1 = np.random.random(N)                                    
x2 = x1 + [np.random.randint(10)/10 for i in range(N)] + b  
C1 = [x1, x2]                                               
x1 = np.random.random(N)
x2 = x1 - [np.random.randint(10)/10 for i in range(N)] - 0.1 + b
C2 = [x1, x2]
print(C1, C2)

# Создаем разделяющую прямую
f = [0+b, 1+b]  

# Задаем весовые кф
w2 = 0.5
w3 = -b*w2
w = np.array([-w2, w2, w3])
# Проводим классификацию двух классов
for i in range(N):
    x = np.array([C1[0][i], C1[1][i], 1])
    y = np.dot(w, x)
    if y >= 0:
        print("Класс C1")
    else:
        print("Класс C2")

plt.scatter(C1[0][:], C1[1][:], s=10, c='red')
plt.scatter(C2[0][:], C2[1][:], s=10, c='blue')
plt.plot(f)
plt.grid(True)
plt.show()
#_______________________________________________________________________________________________
'''
Понятно, что на практике встречаются задачи сложнее. Представим, что наши классы
распределены более сложным образом. В данном случае невозможно
провести одну разделяющую прямую. Проведем две разделяющие прямые. Таким образом
необходимо добавить еще один нейрон. Каждый нейрон будет отвечать за свою прямую.
Затем их классификация объединяется результирующим нейроном выходного слоя.
Представим, что на выходе подаются только значения вершин квадрата.
'''
def act(x):
    return 0 if x <= 0 else 1

def go(C):
    x = np.array([C[0], C[1], 1])
    w1 = [1, 1, -1.5]
    w2 = [1, 1, -0.5]
    w_hidden = np.array([w1, w2])
    w_out = np.array([-1, 1, -0.5])

    sum = np.dot(w_hidden, x)  # вычисляем сумму на входах нейронов скрытого слоя

    out = [act(x) for x in sum]
    out.append(1)
    out = np.array(out)

    sum = np.dot(w_out, out)
    y = act(sum)
    return y

C1 = [[1, 0], [0, 1]]
C2 = [[0, 0], [1, 1]]

print(go(C1[0]), go(C1[1]))
print(go(C2[0]), go(C2[1]))
#_______________________________________________________________________________________________
'''
Пример простой нейронной сети на numpy
'''
# Генерации случайных чисел для инициализации весов
np.random.seed(1)
synaptic_weights = 2 * np.random.random((3, 1)) - 1
print(f"Генерация случайных весов:\nW(1,1)={synaptic_weights[0][0]}\nW(1,2)={synaptic_weights[1][0]}\nW(1,3)={synaptic_weights[2][0]}")
print("-------------------------")

# Вычисление сигмоид функции
def sigmoid(x):
     return 1 / (1 + np.exp(-x))
# Вычисление производной от сигмоид функции
def sigm_deriv(x):
    return x * (1 - x)
# Пропускание входных данных через нейрон и получение предсказания
# Конвертация значений во floats
def run_nn(inputs):
    global synaptic_weights
    inputs = inputs.astype(float)
    output = sigmoid(np.dot(inputs, synaptic_weights))
    return output
    
# Тренировка нейронной сети
def train_nn(training_inputs, training_outputs, training_iterations):
    global synaptic_weights
    for iteration in range(training_iterations):
        # перекачивание данных через нейрон
        output = run_nn(training_inputs)

        # вычисление ошибки через обратное распространение MSE
        error = training_outputs - output
            
        # выполнение корректировки весов
        adjustments = np.dot(training_inputs.T, error * sigm_deriv(output))

        synaptic_weights += adjustments

# создание данных для обучения
training_inputs = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
training_outputs = np.array([[0,1,1,0]]).T

# запуск тренировки нейронной сети 
train_nn(training_inputs, training_outputs, 15000)
print(f"Веса после завершения обучения:\nW(1,1)={synaptic_weights[0][0]}\nW(1,2)={synaptic_weights[1][0]}\nW(1,3)={synaptic_weights[2][0]}")
print("-------------------------")

# получение трех чисел от пользователя
user_inp1 = str(input("Первое число(0 или 1): "))
user_inp2 = str(input("Второе число(0 или 1): "))
user_inp3 = str(input("Третье число(0 или 1): "))

print("Проверка на новых данных: {user_inp1} {user_inp2} {user_inp3}")
print("Предсказание нейронной сети: ")
print(run_nn(np.array([user_inp1, user_inp2, user_inp3])))
#_______________________________________________________________________________________________ 
'''
Построение двухслойной нейронный сети для классификации наборов значений
''' 
#--------------------------------------
#---------- Основное задание ----------
#--------------------------------------
from sklearn.datasets import make_classification


# Сигмоида и ее производная
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x)*(1 - sigmoid(x))

# Задаем начальную выборку
r = 107
X,y = make_classification(n_samples=2000, n_features=5, n_informative=4,n_redundant=1, 
n_repeated=0, n_classes=2, n_clusters_per_class=3,class_sep=4,flip_y=0,weights=[0.5,0.5], 
random_state=r)

# Визуальное представление данных (для анализа dataset`а)
plt.figure(figsize=(8, 8))
plt.subplots_adjust(bottom=0.05, top=0.9, left=0.05, right=0.95)
plt.subplot()
plt.scatter(X[:,2], X[:,3], marker='o', c=y, s=25, edgecolor='black')
plt.xlabel('Набор данных №1')
plt.ylabel('Набор данных №2')
plt.show()

# Добавим bias (+1) для смещения разделяющей прямой
y = np.reshape(y, (len(y), 1))
x = np.zeros((len(X),6))
for i in range(len(X)):
    x[i][0] = X[i][0]
    x[i][1] = X[i][1]
    x[i][2] = X[i][2]
    x[i][3] = X[i][3]
    x[i][4] = X[i][4]
    x[i][5] = 1.0

# Разделение на train и test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=r)

# Генерации случайных чисел для инициализации весов
w0 = 2*np.random.random((3, 2)) - 1
w1 = 2*np.random.random((2, 1)) - 1
print(f"Начальные случайные веса входного слоя:\nW1(1,1)={w0[0][0]}\nW1(1,2)={w0[0][1]}\nW1(2,1)={w0[1][0]}\nW1(2,2)={w0[1][1]}\nW1(3,1)={w0[2][0]}\nW1(3,2)={w0[2][1]}")  
print("-------------------------")
print(f"Начальные случайные веса выходного слоя:\nW2(1,1)={w1[0][0]}\nW2(2,1)={w1[1][0]}")  # \nW2(3,1)={w1[2][0]}
print("-------------------------")

number_epochs = 100000        # Количество эпох
errors = []                   # Массив ошибок, чтобы потом построить график
n = 0.1                       # Скорость обучения (learning rate)    
x_input = X_train[:,[2,3,5]]  # Массив входных данных (В данном случае - 2 класса + bias)
y_result = y_train            # Набор правильных ответов, с которыми нн будет сравнивать свои результаты
accuracy = 0                  # Оценка точности работы нейронной сети
# Процесс обучения
for i in range(number_epochs):
    # Прямое распространение (feed forward)
    layer0 = x_input

    layer1 = sigmoid(np.dot(layer0, w0))
    layer2 = sigmoid(np.dot(layer1, w1))

    # ----------------------
    # По логике на выходной слой тоже надо добавить bias, но тогда почему-то перестают работать размерности матриц
    # tmp = sigmoid(np.dot(layer0, w0)) 
    # layer1 = np.zeros((len(tmp), 3))              # (1340x3) - Добавили bias
    # for i in range(len(layer1)):
    #     layer1[i][0] = tmp[i][0]
    #     layer1[i][1] = tmp[i][1]
    #     layer1[i][2] = 1.0
    # layer2 = sigmoid(np.dot(layer1, w1))
    # ----------------------

    # Обратное распространение (back propagation) с использованием градиентного спуска
    layer2_error = y_result - layer2                                 # 1340x1
    layer2_delta = layer2_error * sigmoid_deriv(np.dot(layer1, w1))  # 1340x1 = 1340x1 * (1340x3 * 3x1)

    layer1_error = layer2_delta.dot(w1.T)                            # 1340x3
    layer1_delta = layer1_error * sigmoid_deriv(np.dot(layer0, w0))  # nan = 1340x3 * (1340x3 * 3x2)

    w1 += layer1.T.dot(layer2_delta) * n
    w0 += layer0.T.dot(layer1_delta) * n

    error = np.mean(np.abs(layer2_error))
    errors.append(error)
    accuracy = (1 - error) * 100
print("========================================")
print(f"Веса входного слоя после обучения:\nW1(1,1)={w0[0][0]}\nW1(1,2)={w0[0][1]}\nW1(2,1)={w0[1][0]}\nW1(2,2)={w0[1][1]}\nW1(3,1)={w0[2][0]}\nW1(3,2)={w0[2][1]}")  
print("-------------------------")
print(f"Веса выходного слоя после обучения:\nW2(1,1)={w1[0][0]}\nW2(2,1)={w1[1][0]}")  # \nW2(3,1)={w1[2][0]}
print("-------------------------")

# Демонстрация полученных результатов
plt.plot(errors)
plt.xlabel('Обучение')
plt.ylabel('Ошибка')
plt.show()
print("Точность нейронной сети " + str(round(accuracy,2)) + "%")
print("========================================")

# Рисование разделяющей прямой
fy1 = [8, 0]
fx1 = [1, 0]

fy2 = [0, 0]
fx2 = [0, -9]

y2=[]
for i in range(2000):
    X1 = np.array([x[i][2], x[i][3], x[i][5]])
    y1=sigmoid(np.dot(np.array(w0[:,0]), X1))
    y3=sigmoid(np.dot(np.array(w0[:,1]), X1))
    y4=sigmoid(w1[0][0]*y1 + w1[1][0]*y3)
    y2.append(y4)

plt.scatter(x[:,2], x[:,3],c=y2)

plt.plot(fx1,fy1)
plt.plot(fx2,fy2)

plt.grid(True)
plt.show()
