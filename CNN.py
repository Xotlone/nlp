# Сверточная нейронная сеть без алгоритма обучения на чистом Python. И да, на NumPy она была-бы и быстрее, и меньше
# Реализована XOTLONE

import random
import math

class CNN:
    def __init__(self, filter_count: int=16, size: tuple=(3, 3), stride: int=1):
        '''Инициализация свёрточной нейронной сети'''

        self.filter_count = filter_count # Думаю, объяснять не надо
        self.size = size # Размер скользящего окна
        self.stride = stride # Его шаг
        self.activation = lambda x: max(x, 0) # ReLU (Выпрямленные линейные блоки)
        self.weights = [[random.random() for w in range(size[0] * size[1])] for fc in range(filter_count)] # Случайные веса для каждого фильтра от 0 до 1

    @staticmethod
    def dim_expand(list: list):
        '''Делает из одномерного списка двумерный'''

        sqrt_len = int(math.sqrt(len(list))) # Берём квадратный корень длинны одномерного списка
        return [[list[x * (y + 1)] for x in range(sqrt_len)] for y in range(sqrt_len)]
    
    @staticmethod
    def dim_remove(list: list):
        '''Делает из двумерного списка одномерный'''

        out = []
        for y in list:
            for x in y:
                out.append(x)
        return out

    @staticmethod
    def downsampling(window, area=2, variant='max'):
        '''Субдискретизация для оптимизации'''

        try:
            window[0][0]
        except TypeError:
            window = CNN.dim_expand(window)
        
        out = []
        for y in range(area): # Перечисляем окно по y
            out.append([])
            for x in range(area): # Перечисляем окно по x от y
                val = []
                for y_slice in range(int(len(window) / area)):
                    y_val = window[(y + 1) * y_slice]
                    for x_slice in range(len(y_val)):
                        try:
                            val.append(window[(y + 1) * y_slice][x_slice]) # Берём все значения из части судбескретизации окна
                        except IndexError:
                            val.append(.0)
                
                if variant == 'max':
                    out[-1].append(max(val)) # Выбираем максимальное
                elif variant == 'avg':
                    out[-1].append(sum(val) / len(val)) # Выбираем среднее
                else:
                    raise NameError(f'Типа субдискретизации "{variant}" нет. Выбирайте "max" максимальное или "avg" среднее')
                    
        out = CNN.dim_remove(out) # Ровняем двумерный список до одномерного
        return out
    
    @staticmethod
    def dropout(filters, coefficient=0.2):
        '''Метод дропаута для продотвращения переобучения'''

        out = []
        for filter in filters:
            rand_indexes = [[random.random() <= coefficient for x in range(len(filter[0]))] for y in range(len(filter))] # Создаем индексы, которые будут заполнены нулями
            out.append([])
            for y, y_rand in zip(filter, rand_indexes):
                out[-1].append([])
                for x, x_rand in zip(y, y_rand): 
                    out[-1][-1].append(.0) if x_rand else out[-1][-1].append(x) # Заполняем случайные индексы нулями
        return out


    def run(self, image, downsampling_area=2, downsampling_variant='max', dropout=0.1):
        '''Полный проход по "ахроматической" (значения диапазоном 0-1) матрице'''

        out = []
        steps = [ # Вычисляем количество шагов
            int((len(image[0]) - self.size[0] + 1) / self.stride), # По x
            int((len(image) - self.size[1] + 1) / self.stride) # По y
        ]
        for filter in range(self.filter_count): # Перечисляем фильтры
            out.append([]) # Добавляем в out пустой фильтр
            for y in range(steps[1]): # Перечисляем шаги по y
                out[-1].append([]) # Добавляем к пустому фильтру вектор, в котором будут хранится значение по x
                for x in range(steps[0]): # Перечисляем шаги по x от позиции y
                    window = [] # Создаём пустое скользящее окно
                    for y_s in range(self.size[1]): # Перечисляем пиксели в окне по y
                        for x_s in range(self.size[0]): # Перечисляем пиксели в окне по x от y
                            try: # Пробуем умножить вес на значение пикселя на картинке для вычисление пикселя в скользящем окне
                                window.append(image[(y + y_s) * self.stride][(x + x_s) * self.stride] * self.weights[filter][x_s * (y_s + 1)])
                            except IndexError: # Если окно вылезает за рамки картинки добавляем 0
                                window.append(.0)
                    window = CNN.downsampling(window, downsampling_area, downsampling_variant) # Проводим субдискретизацию
                    out[-1][-1].append(round(self.activation(sum(window)), 2)) # Добавляем к выходу сумму всех пикселей в окне, проведённую через функцию активации
        
        out = CNN.dropout(out, dropout) # Применяем дропаут
        return out

img_size = (10, 10) # Размер чёрно-белого "изображения"
img = [[random.randint(0, 1) for j in range(img_size[0])] for i in range(img_size[1])] # Заполнение изображения случайными значениями 0-1
#img = [[round(random.random(), 2) for j in range(img_size[0])] for i in range(img_size[1])] # Так-же можно использовать все ахроматические цвета, то есть значения между 0-1
for i in img:
    print(i) # Выводим списки x значений по y из изображения
print()

cnn1 = CNN() # Первый слой свёртки
out1 = cnn1.run(img)
cnn2 = CNN() # Второй слой свёртки
out2 = [cnn2.run(i) for i in out1] # Генерируем 256 фильтров второго слоя свёртки для 16 фильтров первого (16 * 16 = 256 фильтров последнего слоя свёртки 0_0 )


for n, f2 in enumerate(out2): # Перечисляем фильтры последнего слоя свёртки
    print('<{:=^50}>\n'.format(f' {n} ФИЛЬТРЫ ПЕРВОГО СЛОЯ СВЁРТКИ '))
    for f1 in f2: # Перечисляем фильтры первого слоя свёртки от последнего
        for i in f1: # Перечисляем пиксели x по y
            print(i) # Выводим списки x значений по y из изображения
        print()