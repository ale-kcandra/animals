import os
import numpy as np
from PIL import Image
import scipy.special
from my_network import MNIST_Neural_Network

CLASS_NAMES = ["elefante", "farfalla", "mucca", "pecora", "scoiattolo"]  # список животных
IMG_WIDTH = 300  # ширина картинки
IMG_HEIGHT = 200  # высота картинки

def train():
    TRAIN_DIR = r"Animals_recognize-main\Animals_recognize-main\animal-dataset\versions\1\animal_dataset_intermediate\train"  # путь к папке train
    
    # собираем картинки
    files, labels = [], []  # пустые списки для путей и меток
    for label_index, animal in enumerate(CLASS_NAMES):  # перебираем животных
        animal_folder = os.path.join(TRAIN_DIR, f"{animal}_train")  # путь к папке животного
        if not os.path.exists(animal_folder):  # если папки нет — пропускаем
            continue
        for file_name in os.listdir(animal_folder):  # перебираем файлы в папке
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):  # берем только картинки
                files.append(os.path.join(animal_folder, file_name))  # запоминаем путь
                labels.append(label_index)  # запоминаем метку

    print(f"найдено картинок: {len(files)}")  # выводим число найденных картинок

    input_nodes = IMG_WIDTH * IMG_HEIGHT * 3  # 180000 (входной слой)
    net = MNIST_Neural_Network(input_nodes=input_nodes, hidden_nodes=100, output_nodes=len(CLASS_NAMES), learning_rate=0.1)  # создаем сеть
    
    for epoch in range(1):  # пока ставим 1 эпоху для проверки
        print(f"эпоха {epoch+1}/1", end='\r')  # выводим номер эпохи
    
        for i in range(100):  # проходим по 100 картинкам
            img_path = files[i]  # путь к текущей картинке
            label = labels[i]  # номер класса картинки
            
            # 1. принудительно делаем rgb и нужный размер
            img = Image.open(img_path).convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))  # открываем, красим в rgb и меняем размер
            
            # 2. жестко забираем пиксели и делаем плоский массив (h*w*c)
            pixels = np.array(img, dtype=float).ravel()  # получаем плоский список пикселей
            
            # 3. нормализация (как в твоей части)
            pixels = (pixels / 255.0 * 0.99) + 0.01  # нормализуем значения от 0.01 до 1.0
            
            # 4. превращаем в вектор-столбец (180000, 1)
            inputs = pixels.reshape(input_nodes, 1)  # делаем из плоского списка вектор-столбец
            
            # 5. создаем правильный ответ
            targets = np.zeros(len(CLASS_NAMES)) + 0.01  # заполняем нулями плюс 0.01
            targets[label] = 0.99  # ставим 0.99 на правильный ответ
            targets = targets.reshape(len(CLASS_NAMES), 1)  # делаем вектор-столбец
            
            # 6. прямой проход
            hidden_outputs = net.activation_function(np.dot(net.weights_input_hidden, inputs))  # считаем выход скрытого слоя
            final_outputs = net.activation_function(np.dot(net.weights_hidden_output, hidden_outputs))  # считаем выход сети
            
            # 7. обратный проход
            output_errors = targets - final_outputs  # считаем ошибку на выходе
            hidden_errors = np.dot(net.weights_hidden_output.T, output_errors)  # считаем ошибку на скрытом слое
            net.weights_hidden_output += net.learning_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))  # обновляем веса скрытый-выход
            net.weights_input_hidden += net.learning_rate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))  # обновляем веса вход-скрытый
    
    np.savez("my_weights.npz", w1=net.weights_input_hidden, w2=net.weights_hidden_output)  # сохраняем веса в файл

if __name__ == "__main__":
    train()  # запускаем обучение