import os
import pandas as pd
import numpy as np
import time
import cv2
import tensorflow as tf
import keras.backend as K
import keras
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score,precision_score,f1_score,recall_score, make_scorer
from sklearn import preprocessing
from sklearn import svm

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import plot_confusion_matrix

import matplotlib.pyplot as plt

import pickle
import joblib
from joblib import dump, load

from collections import Counter
from imblearn.over_sampling import SMOTE 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import datasets, layers, models


#---------------------------------------------------------------------DATA PROCESSING--------------------------------------------------------------
def file_name_encoding_init(file_names):
    le_filename = preprocessing.LabelEncoder()
    le_filename.fit(file_names)
    dump(le_filename, 'Task A Assets/Task_A_file_name_encoder.joblib') 


def SMOTE_preprocessing(file_names, labels):
    le_filename = joblib.load('Task A Assets/Task_A_file_name_encoder.joblib')
    le_filename.fit(file_names)
    le_file_names = le_filename.transform(file_names).reshape(-1,1)
    
    sm = SMOTE()
    SMOTE_file_names, SMOTE_labels = sm.fit_resample(le_file_names, labels)
    print("Distribution of image categories post-SMOTE:")
    print(Counter(list(SMOTE_labels)))
    
    SMOTE_file_names = le_filename.inverse_transform(SMOTE_file_names)

    return SMOTE_file_names, SMOTE_labels

def PCA_process(x):
    pca = KernelPCA(kernel='rbf', n_components=800)
    x_flat = np.array([features_2d.flatten() for features_2d in x])
    x_pca = pca.fit_transform(x_flat)
    dump(pca, 'Task A Assets/Task_A_PCA.joblib') 
    return x_pca


def image_processing(data_path,file_names,model):
    dataset_tumor=[]
    if model == 'svm':
        for file_name in file_names:
            file=cv2.imread(data_path+"/image/"+file_name)
            file_resize=cv2.resize(file,(256,256))
            dataset_tumor.append(file_resize)
        tumor_data = np.array(dataset_tumor)
        tumor_data = PCA_process(tumor_data)
            
    elif model == 'cnn':
        for file_name in file_names:
            file=cv2.imread(data_path+"/image/"+file_name)
            file_resize=cv2.resize(file,(128,128))/255.
            dataset_tumor.append(file_resize)
        tumor_data = np.array(dataset_tumor)
    return tumor_data

def preprocessing_data(data_path, file, status, task, model):

    data=pd.read_csv(data_path+file)
    file_names=list(data['file_name'])
    
    if task == 'task_a':
        data['label'] = data['label'].apply(lambda x: "no_tumor" if x == "no_tumor" else "tumor")
        
    labels=data['label'].values.ravel()
    
    if status == 'training':
        print("Distribution of image categories:")
        print(Counter(list(data['label'])))
        
        file_name_encoding_init(file_names)
        
        file_names, labels = SMOTE_preprocessing(file_names, labels)
        
        x = image_processing(data_path, file_names, model)
        x_train,x_test,y_train,y_test = train_test_split(x,labels,test_size=0.2)
        
        if model == 'svm':
            le_label = joblib.load('Task A Assets/Task_A_label_encoder.joblib')
            y_train = le_label.transform(y_train)
            y_test = le_label.transform(y_test)
            
        elif model == 'cnn':   
            ohe = OneHotEncoder(handle_unknown = "ignore", sparse=False)
            ohe = ohe.fit(labels.reshape(-1,1))
            y_train = ohe.transform(np.array(y_train).reshape(-1,1))
            y_test = ohe.transform(np.array(y_test).reshape(-1,1))
            if task == 'task_a':
                dump(ohe, 'Task A Assets/Task_A_one-hot_encoder.joblib') 
            elif task == 'task_b':
                dump(ohe, 'Task B Assets/Task_B_one-hot_encoder.joblib') 
    
                    
        return x_train, x_test, y_train, y_test

    
    elif status == 'testing':  
        if model == 'svm':
            encoder = joblib.load('Task A Assets/Task_A_label_encoder.joblib')
        elif model == 'cnn':
            if task == 'task_a':
                encoder = joblib.load('Task A Assets/Task_A_one-hot_encoder.joblib')
            elif task == 'task_b':
                encoder = joblib.load('Task B Assets/Task_B_one-hot_encoder.joblib')
        
        x_test = image_processing(data_path,file_names, model) 
        y_test = encoder.transform(labels)
        
        return x_test, y_test

#---------------------------------------------------------------------HYPERPARAMETER PROCESSING--------------------------------------------------------------
def find_SVM_params(x_train, y_train, x_test, y_test):
    classifiers=[svm.SVC()]
    classifierNames=['SVM']
    parameters=[{'kernel':['rbf', 'sigmoid', 'poly'],'C':[0.7,2,10]}]

    for i in range(len(classifiers)):
        clf=classifiers[i]
        clf_search= GridSearchCV(clf, parameters[i], scoring = 'accuracy',cv = 3)
        result = clf_search.fit(x_train,y_train)
        best_clf=clf_search.best_estimator_
        
        print("Classifier:",classifierNames[i])
        print("Best Parameters: {}".format(clf_search.best_params_))
        print("Best Validation Accuracy: %f" % (result.best_score_))

        pred=best_clf.predict(x_test)

        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test,pred)
        recall = recall_score(y_test,pred)
        f1score = f1_score(y_test,pred)
        
        print("Classifier Performance:",classifierNames[i])
        print("Test Accuracy: %.4f" %(accuracy))
        print("Test Precision: %.4f" %(precision))
        print("Test Recall: %.4f" %(recall))
        print("Test F1-score: %.4f" %(f1score))
    
    joblib.dump(clf_search, 'Task A Assets/gridsearchcv_svm.pkl')
    
def define_CNN(task):
    if task == 'task_a':
        output_nodes = 2
    elif task == 'task_b':
        output_nodes = 4
    
    model = Sequential()
    
    #Conv2D is a 2D convolutional layer that applies 2d convolution to input signal composed of several input planes.
    #Kernel: is the filter that moves over the input layer to obtain a matrix of dot products based on the kernel size.
    #Strides metric: determines the shift of the kernel filter as it convolves around the input volume. 
    #Higher stride means larger shifts which lead to a smaller output volume. 
    #It is important to for the stride to be small enough that we capture significant features but not too small as this could lead to overfitting.
    #Padding: refers to the amount of pixels added to an image when it is being processed by the kernel. 
    #This helps provide more space for the kernel to cover the image.
    #Activation: refers to activation function used to product output layer. Here I use ReLU as it is simple and doesn't suffer from vanishing gradients
    
    model.add(Conv2D(32, kernel_size=(31,31), padding='same', activation='relu', input_shape=(128,128,3)))
    #MaxPooling reduces the size of the output matrix by obtaining the largest value from the a given pooling matrix 
    model.add(MaxPooling2D(pool_size=(2,2)))
    #Batch Normalization helps reduce the internal covariate shift of the network. 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, kernel_size=(27,27), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
              
    model.add(Conv2D(128, kernel_size=(25,25), strides=(2,2), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    
    #Flatten layer converts data into a 1-dimensional array for inputting it to the next layer. 
    #We flatten the output of the convolutional layers to create a single long feature vector 
    #So basically after our layers of pooling, we want to extract the data so we can feed it into our neural network
    model.add(Flatten())
    
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(output_nodes, activation='softmax'))
    model.compile(loss="categorical_crossentropy",optimizer=Adam(learning_rate=0.001),  metrics=['acc'])
    
    return model    

def find_CNN_params(x_train, y_train, x_test, y_test, task):
    clf = KerasClassifier(build_fn=lambda: define_CNN(task))
    params={'batch_size':[100, 50, 32], 
            'nb_epoch':[10, 25, 50]}
    scorers = {
        'accuracy_score': make_scorer(accuracy_score)
        }
    clf_search=GridSearchCV(estimator=clf, param_grid=params, cv=3, scoring=scorers, refit="accuracy_score")
    result = clf_search.fit(x_train,y_train)
    best_clf=clf_search.best_estimator_

    print("Classifier: CNN - ", task)
    print("Best Parameters: {}".format(clf_search.best_params_))
    print("Best Validation Accuracy: %f" % (result.best_score_))

    pred=best_clf.predict(x_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test,pred)
    recall = recall_score(y_test,pred)
    f1score = f1_score(y_test,pred)

    print("Classifier Performance: CNN - ", task)
    print("Test Accuracy: %.4f" %(accuracy))
    print("Test Precision: %.4f" %(precision))
    print("Test Recall: %.4f" %(recall))
    print("Test F1-score: %.4f" %(f1score))
    
    if task == 'task_a':
        joblib.dump(clf_search, 'Task A Assets/gridsearchcv_cnn.pkl')
    elif task == 'task_b':
        joblib.dump(clf_search, 'Task B Assets/gridsearchcv_cnn.pkl')
#---------------------------------------------------------------------TRAINING PROCESSING--------------------------------------------------------------



def train_SVM(x_train_val,y_train_val):
    
    kf = KFold(n_splits=3,shuffle=True)
    
    val_accuracy = []
    val_precision = []
    val_recall = []
    val_f1score = []
    
    
    print("SVM training with 3-Fold Cross Validation.")
    for train_index, test_index in kf.split(x_train_val):
        x_train, x_val = x_train_val[train_index], x_train_val[test_index]
        y_train, y_val = y_train_val[train_index], y_train_val[test_index]

        model=svm.SVC(C=0.7,kernel='rbf')
        model.fit(x_train,y_train)

        pred_val=model.predict(x_val)

        val_accuracy.append(accuracy_score(y_val, pred_val))
        val_precision.append(precision_score(y_val, pred_val))
        val_recall.append(f1_score(y_val, pred_val))
        val_f1score.append(recall_score(y_val, pred_val))
        

    average_val_accuracy=sum(val_accuracy)/len(val_accuracy)
    average_val_precision=sum(val_precision)/len(val_precision)
    average_val_recall=sum(val_recall)/len(val_recall)
    average_val_f1score=sum(val_f1score)/len(val_f1score)
    

    print("SVM Classifier 3-Fold CV:")
    print("Average Acc: %.4f" %(average_val_accuracy))
    print("Average Precision: %.4f" %(average_val_precision))
    print("Average recall: %.4f" %(average_val_recall))
    print("Average F1 Score: %.4f \n" %(average_val_f1score))
    

    pickle.dump(model, open("Task A Assets/Task_A_SVM_Model", 'wb'))

def train_CNN(x_train_val, y_train_val, task):
    if task == 'task_a':
        ohe = joblib.load('Task A Assets/Task_A_one-hot_encoder.joblib')
        ohe_depth = 2
    elif task == 'task_b':
        ohe = joblib.load('Task B Assets/Task_B_one-hot_encoder.joblib')
        ohe_depth = 4
    kf = KFold(n_splits=5,shuffle=True)
    k_number = 0
    
    val_accuracy = []
    val_precision = []
    val_recall = []
    val_f1score = []
    
    
    print("CNN training with 5-Fold Cross Validation.")
    for train_index, test_index in kf.split(x_train_val):
        k_number += 1
        x_train, x_val = x_train_val[train_index], x_train_val[test_index]
        y_train, y_val = y_train_val[train_index], y_train_val[test_index]
        
        model = define_CNN(task)
        
        
        history1 = model.fit(x_train,y_train,epochs=50,batch_size=32,shuffle=True,validation_split=0.1)
        
        acc_history = history1.history['acc']
        val_acc_history = history1.history['val_acc']
        loss_history = history1.history['loss']
        val_loss_history = history1.history['val_loss']
        
        plt.plot(history1.history['acc'])
        plt.plot(history1.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history1.history['loss'])
        plt.plot(history1.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


        print("The highest validation acc is {}".format(np.max(val_acc_history)))

        result=model.predict(x_val)
        result_class = tf.one_hot(np.argmax(result, axis=1), depth = ohe_depth)
        
        result_class = ohe.inverse_transform(result_class)
        y_val_class = ohe.inverse_transform(y_val)

        val_accuracy.append(accuracy_score(result_class, y_val_class))
        val_precision.append(precision_score(result_class, y_val_class,average='micro'))
        val_f1score.append(f1_score(result_class, y_val_class,average='micro'))
        val_recall.append(recall_score(result_class, y_val_class,average='micro'))

        average_val_accuracy=sum(val_accuracy)/len(val_accuracy)
        average_val_precision=sum(val_precision)/len(val_precision)
        average_val_recall=sum(val_recall)/len(val_recall)
        average_val_f1score=sum(val_f1score)/len(val_f1score)
        
        print("CNN 5-Fold CV:")
        print("Average Acc: %.4f" %(average_val_accuracy))
        print("Average Precision: %.4f" %(average_val_precision))
        print("Average recall: %.4f" %(average_val_recall))
        print("Average F1 Score: %.4f \n" %(average_val_f1score))
        
    if task == 'task_a':
        model.save('Task A Assets/Task_A_CNN_Model')  

    elif task == 'task_b':
        model.save('Task B Assets/Task_B_CNN_Model')
#---------------------------------------------------------------------TESTING PROCESSING--------------------------------------------------------------
def test_SVM(x_test, y_test):

    loaded_model = pickle.load(open("Task A Assets/Task_A_SVM_Model", 'rb'))
    y_pred_svm = loaded_model.predict(x_test)
    print('Accuracy on test set: '+str(accuracy_score(y_test,y_pred_svm)))
    
    #text report showing the main classification metrics
    print(classification_report(y_test,y_pred_svm))
    plt.figure(figsize = (5,5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_svm, cmap = 'Blues')
    plt.show()

def test_CNN(x_test, y_test, task):
    if task == 'task_a':
        loaded_model = keras.models.load_model('Task A Assets/Task_A_CNN_Model')
        ohe = joblib.load('Task A Assets/Task_A_one-hot_encoder.joblib')
        ohe_depth = 2
    elif task == 'task_b':
        loaded_model = keras.models.load_model('Task B Assets/Task_B_CNN_Model')
        ohe = joblib.load('Task B Assets/Task_B_one-hot_encoder.joblib')
        ohe_depth = 4
        
    result=loaded_model.predict(x_test)
    result_class = tf.one_hot(np.argmax(result, axis=1), depth = ohe_depth)

    result_class = ohe.inverse_transform(result_class)
    y_test_class = ohe.inverse_transform(y_test)
    
    acc = accuracy_score(result_class, y_test_class)
    print("Accuracy for test data:", acc)
    
    plt.figure(figsize = (7,7))
    ConfusionMatrixDisplay.from_predictions(y_test_class, result_class, cmap = 'Blues')
    plt.xticks(rotation=45)
    plt.show()
    print(loaded_model.summary())
    print(classification_report(y_test_class, result_class))  