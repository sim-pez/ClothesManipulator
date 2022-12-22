from cProfile import label
from operator import truediv
from tqdm.auto import tqdm
import constants as C
import random
import torchvision.transforms as transforms
import numpy as np
import h5py
from dataloader import Data
from pprint import pprint
from joblib import Parallel, delayed
import multiprocessing
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle 

def create_models():

    file_root = 'splits/Shopping100k'
    img_root_path = '/Users/simone/Desktop/VMR/Dataset/Shopping100k/Images'

    print('Loading attributes')
    train_data = Data(file_root,  img_root_path, 
                          transforms.Compose([
                              transforms.Resize((C.TRAIN_INIT_IMAGE_SIZE, C.TRAIN_INIT_IMAGE_SIZE)),
                              transforms.RandomHorizontalFlip(),
                              transforms.CenterCrop(C.TARGET_IMAGE_SIZE),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                          ]), 'train')
    
    labels = train_data.label_data
    attr_num = train_data.attr_num
    features_data = np.load(f'eval_out/feat_train.npy')

    print('creating models')

    MinMaxScaler = preprocessing.MinMaxScaler()
    X_data_minmax = MinMaxScaler.fit_transform(features_data)
    with open(f'knn/minmaxscaler', 'wb') as pickle_file:
            pickle.dump(MinMaxScaler, pickle_file)
    y_datas = []
    offset = 0
    for a in attr_num:
        y_datas.append([label[offset : a+offset] for label in labels])
        offset += a

    for attribute_number, y_data in tqdm(enumerate(y_datas)):
        knn_clf=KNeighborsClassifier(n_neighbors=5, metric='minkowski', algorithm='ball_tree', n_jobs = -1) #0.88 accuracy
        knn_clf.fit(X_data_minmax, y_data)
        with open(f'knn/knn_model_att_{attribute_number}', 'wb') as pickle_file:
            pickle.dump(knn_clf, pickle_file)

    print('done!')
    

def get_knn_hyperparams(X_train, y_train, attribute):

    file_root = 'splits/Shopping100k'
    img_root_path = '/Users/simone/Desktop/VMR/Dataset/Shopping100k/Images'

    print('Loading attributes')
    train_data = Data(file_root,  img_root_path, 
                          transforms.Compose([
                              transforms.Resize((C.TRAIN_INIT_IMAGE_SIZE, C.TRAIN_INIT_IMAGE_SIZE)),
                              transforms.RandomHorizontalFlip(),
                              transforms.CenterCrop(C.TARGET_IMAGE_SIZE),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                          ]), 'train')

    labels = train_data.label_data
    attr_num = train_data.attr_num
    features_data = np.load(f'eval_out/feat_train.npy')

    print('finding hyperparameters')

    MinMaxScaler = preprocessing.MinMaxScaler()
    X_data_minmax = MinMaxScaler.fit_transform(features_data)
    y_datas = []
    offset = 0
    for a in attr_num:
        y_datas.append([label[offset : a+offset] for label in labels])
        offset += a

    for attribute_number, y_data in enumerate(y_datas):
        X_train, X_test, y_train, y_test = train_test_split(X_data_minmax, y_data, test_size=0.2, random_state = 1)
        get_knn_hyperparams(X_train, y_train, attribute_number)

    with open("knn/parameters.txt", "a") as file:
        file.write("attribute number " + str(attribute) + "\n")

    #List Hyperparameters to tune
    hyperparameters = {
    'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev']
    }   
    knn_clf=KNeighborsClassifier()
    clf = GridSearchCV(knn_clf, hyperparameters, cv=4, verbose=10, n_jobs=-1)
    best_model = clf.fit(X_train,y_train) 
    print('Best metric:', best_model.best_estimator_.get_params()['metric'])
    with open("knn/parameters.txt", "a") as file:
        file.write('Best metric:' + str(best_model.best_estimator_.get_params()['metric']) + "\n")

    hyperparameters = {
    'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],          
    }   
    knn_clf=KNeighborsClassifier()
    clf = GridSearchCV(knn_clf, hyperparameters, cv=4, verbose=10, n_jobs=-1)
    best_model = clf.fit(X_train,y_train) 
    with open("knn/parameters.txt", "a") as file:
        file.write('Best algorithm:' + str(best_model.best_estimator_.get_params()['algorithm']) + "\n")

    hyperparameters = {
    'n_neighbors': [2, 3, 5, 7, 10],                                   
    }   
    knn_clf=KNeighborsClassifier()
    clf = GridSearchCV(knn_clf, hyperparameters, cv=4, verbose=10, n_jobs=-1)
    best_model = clf.fit(X_train,y_train) 
    with open("knn/parameters.txt", "a") as file:
        file.write('Best n_neighbors:'+ str(best_model.best_estimator_.get_params()['n_neighbors']) + "\n\n")



def calculate_accuracy(labels, predictions):

    different_arrays = 0

    for actual_array, prediction_array in zip(labels, predictions):
        for actual, prediction in zip(actual_array, prediction_array):
            if actual != prediction:
                different_arrays += 1 
                break
       
    array_acc = 1 - (different_arrays / float(len(labels)))
    print(f"Completely same arrays: {array_acc}")


    split_idx = np.load('multi_manip/split_index.obj', allow_pickle = True)
    split_idx = [i - 1 for i in split_idx]
    attr_cnt = 0
    error_cnt = 0
    for actual_array, prediction_array in zip(labels, predictions):
        found_error = False
        for i, (actual, prediction) in enumerate(zip(actual_array, prediction_array)):
            if actual != prediction:
                found_error = True
            if i in split_idx:
                if found_error:
                    error_cnt += 1
                    found_error = False
                attr_cnt += 1

    attr_acc = 1 - error_cnt / float(attr_cnt)
    
    print(f"Attributes accuracy: {attr_acc}")

    

class KNNModel():

    def __init__(self):
        self.knn_models = []
        for attribute_number in range(len(attr_num)):
            self.knn_models.append(pickle.load(open(f'knn/knn_model_att_{attribute_number}', 'rb')))
        self.minMaxScaler = pickle.load(open(f'knn/minmaxscaler', 'rb'))
    
    def predict(self, features_list):
        
        features_list = self.minMaxScaler.transform(features_list)
        
        predictions = []
        for model in self.knn_models:
            predictions.append(model.predict(features_list))    

        flattened_predictions = [[] for _ in range(len(features_list))]
        for model_predictions in predictions:
            for i, model_pred in enumerate(model_predictions):
                flattened_predictions[i].extend(model_pred)

        for i, prediction in enumerate(flattened_predictions):
            flattened_predictions[i] = np.array(prediction)

        return flattened_predictions


if __name__ == '__main__':

    file_root = 'splits/Shopping100k'
    img_root_path = '/Users/simone/Desktop/VMR/Dataset/Shopping100k/Images'

    test_data = Data(file_root,  img_root_path,
                      transforms.Compose([
                          transforms.Resize((C.TARGET_IMAGE_SIZE, C.TARGET_IMAGE_SIZE)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=C.IMAGE_MEAN, std=C.IMAGE_STD)
                      ]), 'test')
    
    attr_num = test_data.attr_num
    labels = test_data.label_data

    features_list = np.load(f'eval_out/feat_test.npy')

    knn = KNNModel()

    predictions = knn.predict(features_list)

    calculate_accuracy(labels, predictions)

    #TODO scrivi funzione find_nearest_image(feature)