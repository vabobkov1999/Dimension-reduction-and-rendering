from sklearn.preprocessing import LabelEncoder
import numpy as np
import mahotas
import cv2
import os
import h5py

fixed_size = tuple((500, 500))
train_path = "dataset/train"
h5_data = 'data_LR3.h5'
h5_labels = 'labels_LR3.h5'
bins = 8

#Используем гистограмму для определения фичи - цвета
def histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

#Используем HU_MOMENTS для определения фичи - контуров
def hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

#Используем ХАРАЛИК для определения фичи - текстуры
def haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

#Начинаем обучение
train_labels = os.listdir(train_path)
train_labels.sort()
print(train_labels)

global_features = []
labels = []

for training_name in train_labels:
    dir = os.path.join(train_path, training_name)
    current_label = training_name

    path = train_path + "/" + training_name
    x = os.listdir(dir)

    for y in x:
        file = path + "/" + str(y)

        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)

        f_histogram = histogram(image)
        f_hu_moments = hu_moments(image)
        f_haralick = haralick(image)

        global_feature = np.hstack([f_histogram, f_hu_moments, f_haralick])

        labels.append(current_label)
        global_features.append(global_feature)

    print("[STATUS] processed folder: {}".format(current_label))

print("[STATUS] completed Global Feature Extraction...")

print("[STATUS] feature vector size {}".format(np.array(global_features).shape))
print("[STATUS] training Labels {}".format(np.array(labels).shape))

targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print("[STATUS] training labels encoded...")

print("[STATUS] target labels: {}".format(target))
print("[STATUS] target labels shape: {}".format(target.shape))

h5f_data = h5py.File(h5_data, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(global_features))

h5f_label = h5py.File(h5_labels, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print("[STATUS] end of training..")
#Заканчиваем обучение
