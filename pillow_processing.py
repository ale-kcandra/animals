import os
import numpy as np
from PIL import Image

def input_from_picture(picture_path):
    img = Image.open(picture_path).convert("RGB")  # открываем картинку по пути
    pixels = list(img.get_flattened_data())  # получаем все пиксели одним списком
    pixels = (np.asarray(pixels, dtype = float) / 255.0 * 0.99) + 0.01  # нормализуем значения в диапазон 0.01..1.0
    return pixels  # возвращаем массив пикселей

# тут задаем классы прямо в этом файле, чтобы не тянуть model.py
CLASS_NAMES = ["elefante", "farfalla", "mucca", "pecora", "scoiattolo"]  # список животных 
IMG_WIDTH = 300  # ширина картинки
IMG_HEIGHT = 200  # высота картинки

#обработка картинок
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")  # открываем и переводим в RGB
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)  # приводим все картинки к одному размеру
    img.save("_tmp.jpg", quality=90)  # сохраняем во временный файл, чтобы твоя функция его прочитала
    pixels = input_from_picture("_tmp.jpg")  # твоя функция
    return pixels  # возвращаем массив numpy

#подает весь датасет для обучения
class AnimalDataset:
    def __init__(self, train_folder):
        self.files = []  # список путей к картинкам
        self.labels = []  # список меток классов
        for label_index, animal in enumerate(CLASS_NAMES):  # обходим папки с каждым животным
            animal_folder = os.path.join(train_folder, f"{animal}_train")  # путь к папке животного
            if not os.path.exists(animal_folder):  # если папки нет, то пропускаем
                continue
            for file_name in os.listdir(animal_folder):  # перебираем файлы в папке
                if file_name.lower().endswith((".jpg", ".jpeg", ".png")):  # берём только картинки
                    full_path = os.path.join(animal_folder, file_name)  # полный путь к файлу
                    self.files.append(full_path)  # запоминаем путь
                    self.labels.append(label_index)  # запоминаем метку

    def __len__(self):
        return len(self.files)  # количество картинок в датасете

    def __getitem__(self, idx):
        file_path = self.files[idx]  # путь к картинке по индексу
        label = self.labels[idx]  # метка класса по индексу
        image_array = load_image(file_path)  # загружаем картинку и превращаем в массив
        return image_array, label  # возвращаем массив пикселей и метку

# проверка, что всё собирается
if __name__ == "__main__":
    TRAIN_DIR = r"Animals_recognize-main\Animals_recognize-main\animal-dataset\versions\1\animal_dataset_intermediate\train"  # путь к папке train
    dataset = AnimalDataset(TRAIN_DIR)  # создаем датасет
    print("Найдено картинок:", len(dataset))  # печатаем количество найденных картинок
    if len(dataset) > 0:  # если картинки есть
        img, label = dataset[0]  # берем первую картинку
        print("Размер массива (чисел):", img.shape)  # печатаем размер массива
        print("Класс (номер):", label, "->", CLASS_NAMES[label])  # печатаем метку и название класса