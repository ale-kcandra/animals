import numpy as np
import scipy.special

class MNIST_Neural_Network:
    def __init__(self, input_nodes=3, hidden_nodes=3,
                 output_nodes=3, learning_rate=0.5, epochs=5):
        self.input_nodes = input_nodes  # количество нейронов на входе (у нас 180000 пикселей)
        self.hidden_nodes = hidden_nodes  # количество нейронов в скрытом слое
        self.output_nodes = output_nodes  # количество нейронов на выходе (количество животных)
        self.learning_rate = learning_rate  # скорость обучения (шаг изменения весов)
        self.epochs = epochs  # количество эпох обучения
        
        # Инициализация весов случайными числами с нормальным распределением
        # Матрица весов от входного слоя к скрытому: размер (hidden_nodes, input_nodes)
        self.weights_input_hidden = np.random.normal(0.0, pow(self.input_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        # Матрица весов от скрытого слоя к выходному: размер (output_nodes, hidden_nodes)
        self.weights_hidden_output = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.output_nodes, self.hidden_nodes))
        
        # Функция активации это наша сигмоида (expit — это быстрая версия 1 / (1 + exp(-x)))
        self.activation_function = lambda x: scipy.special.expit(x)

    def query(self, inputs_list):
        # превращаем список входных данных в вертикальный вектор-столбец
        inputs = np.array(inputs_list, ndmin=2).T
        # считаем сигналы на скрытом слое: умножаем веса на входы и применяем сигмоиду
        hidden_input = np.dot(self.weights_input_hidden, inputs)
        hidden_output = self.activation_function(hidden_input)
        # считаем сигналы на выходном слое: умножаем веса на выход скрытого слоя и применяем сигмоиду
        last_input = np.dot(self.weights_hidden_output, hidden_output)
        last_output = self.activation_function(last_input)
        
        # возвращаем итоговые значения (вероятности для каждого животного)
        return last_output