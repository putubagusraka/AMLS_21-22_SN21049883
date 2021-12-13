import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import keras

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,precision_score,f1_score,recall_score
from sklearn import preprocessing
from sklearn import svm

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import KFold


import matplotlib.pyplot as plt

import pickle
import joblib
from joblib import dump, load


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, SGD


#---------------------------------------------------------------------TRAINING PROCESSING--------------------------------------------------------------
def ablation_train_SVM(x_train_val,y_train_val):
    """
    This is the SVM training function with 5-fold cross validation for ablation studies, changed hyperparameter.
    Each 5 split data is trained with SVM model
    Here accuracy, precision, recall, and f1 score are tallied and average for the 5 models
    The SVM model is saved to local directory for any further testing
    
    Key Parameters:
    x_train_val = the training set as result from the preprocessing function, split into training and validation data 
    y_train_val = the supervised labels for training set as result from the preprocessing function, split into training and validation data 
    
    Returns:
    SVM Model with different hyperparameters is saved to local directory to be called freely
    """
    
    kf = KFold(n_splits=5,shuffle=True)
    
    val_accuracy = []
    val_precision = []
    val_recall = []
    val_f1score = []
    
    
    print("SVM training with 5-Fold Cross Validation.")
    for train_index, test_index in kf.split(x_train_val):
        x_train, x_val = x_train_val[train_index], x_train_val[test_index]
        y_train, y_val = y_train_val[train_index], y_train_val[test_index]

        model=svm.SVC(C=5,kernel='rbf')
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
    

    print("SVM Classifier 5-Fold CV:")
    print("Average Acc: %.4f" %(average_val_accuracy))
    print("Average Precision: %.4f" %(average_val_precision))
    print("Average recall: %.4f" %(average_val_recall))
    print("Average F1 Score: %.4f \n" %(average_val_f1score))
    

    pickle.dump(model, open("Ablation Assets/Ablation_SVM_Model", 'wb'))

def ablation_define_CNN(task):
    """
    This function returns the CNN architecture used for task A and task B.
    #Conv2D is a 2D convolutional layer that applies 2d convolution to input signal composed of several input planes.
    #Kernel: is the filter that moves over the input layer to obtain a matrix of dot products based on the kernel size.
    #Strides metric: determines the shift of the kernel filter as it convolves around the input volume. 
    #Higher stride means larger shifts which lead to a smaller output volume. 
    #It is important to for the stride to be small enough that we capture significant features but not too small as this could lead to overfitting.
    #Padding: refers to the amount of pixels added to an image when it is being processed by the kernel. 
    #This helps provide more space for the kernel to cover the image.
    #Activation: refers to activation function used to product output layer. Here I use ReLU as it is simple and doesn't suffer from vanishing gradients
    #MaxPooling reduces the size of the output matrix by obtaining the largest value from the a given pooling matrix 
    #Batch Normalization helps reduce the internal covariate shift of the network.
    #Flatten layer converts data into a 1-dimensional array for inputting it to the next layer. 
    #We flatten the output of the convolutional layers to create a single long feature vector 
    #So basically after our layers of pooling, we want to extract the data so we can feed it into our neural network
    
    Key Parameters:
    task = task_a or task_b, the only distinction between task a and b is the output node count.
    
    Returns:
    model = the CNN model, ready to use for model training
    
    """
    
    if task == 'task_a':
        output_nodes = 2
    elif task == 'task_b':
        output_nodes = 4
    
    model = Sequential()
    
    
    model.add(Conv2D(64, kernel_size=(5,5), padding='same', activation='relu', input_shape=(128,128,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(BatchNormalization())

              
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(BatchNormalization())

    
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(BatchNormalization())

        
    model.add(Conv2D(256, kernel_size=(2,2), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    model.add(BatchNormalization())

       

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(output_nodes, activation='softmax'))
    
    model.compile(loss="categorical_crossentropy",optimizer=Adam(learning_rate=0.001),  metrics=['acc'])
    
    return model   


def ablation_train_CNN(x_train_val, y_train_val, task):
    """
    This is the CNN training function with 5-fold cross validation.
    Each 5 split data is trained with SVM model
    Here accuracy, precision, recall, and f1 score are tallied and average for the 5 models
    The CNN model is saved to local directory for any further testing
    
    Key Parameters:
    x_train_val = the training set as result from the preprocessing function, split into training and validation data 
    y_train_val = the supervised labels for training set as result from the preprocessing function, split into training and validation data 
    task = task_a or task_b, the only distinction between task a and b is the output node count and the argmax function parameter
    
    Returns:
    CNN Model is saved to local directory to be called freely, for either task A and task B
    """
    
    
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
        x_train, x_val = x_train_val[train_index], x_train_val[test_index]
        y_train, y_val = y_train_val[train_index], y_train_val[test_index]
        
        model = ablation_define_CNN(task)
        
        history1 = model.fit(x_train,y_train,epochs=50,batch_size=32, validation_split = 0.1)
        
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
        val_precision.append(precision_score(result_class, y_val_class,average='macro'))
        val_f1score.append(f1_score(result_class, y_val_class,average='macro'))
        val_recall.append(recall_score(result_class, y_val_class,average='macro'))

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
        model.save('Ablation Assets/Ablation_Task_A_CNN_Model')  

    elif task == 'task_b':
        model.save('Ablation Assets/Ablation_Task_B_CNN_Model')

        
#---------------------------------------------------------------------TESTING PROCESSING--------------------------------------------------------------        
def ablation_test_SVM(x_test, y_test):
    """
    The main ablation SVM testing function.
    The function calls the trained model of the SVM that was saved locally, and uses it to test on testing data.
    Results will include an accuracy score, confusion matrix, and a report regarding detailed accuracy, precision, recall, and f1 score
    
    Key Parameters:
    x_test = the testing set used for testing the trained SVM model, must already be processed by the image_processing function
    y_test = supervised labels for  the testing set used for testing the trained SVM model
    
    
    Returns:
    Predicted results for testing data
    Performance metrics of model for testing data 
    """
    loaded_model = pickle.load(open("Ablation Assets/Ablation_SVM_Model", 'rb'))
    y_pred_svm = loaded_model.predict(x_test)
    
    
    #le = joblib.load('Task A Assets/Task_A_label_encoder.joblib')
    #y_test = le.inverse_transform(y_test)
    #y_pred_svm =  le.inverse_transform(y_pred_svm)
    
    print('Accuracy on test set: '+str(accuracy_score(y_test,y_pred_svm)))
    
    #text report showing the main classification metrics
    print(classification_report(y_test,y_pred_svm))
    plt.figure(figsize = (5,5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_svm, cmap = 'Blues')
    plt.show()

def ablation_test_CNN(x_test, y_test, task):
    """
    The main ablation CNN testing function.
    The function calls the trained model of the CNN that was saved locally, and uses it to test on testing data.
    Results will include an accuracy score, confusion matrix, and a report regarding detailed accuracy, precision, recall, and f1 score
    
    Key Parameters:
    x_test = the testing set used for testing the trained CNN model, must already be processed by the image_processing function
    y_test = supervised labels for  the testing set used for testing the trained CNN mode
    task = task_a or task_b, the only distinction between task a and b is the output node count and the argmax function parameter
    
    Returns:
    Predicted results for testing data
    Performance metrics of model for testing data 
    """
    
    if task == 'task_a':
        loaded_model = keras.models.load_model('Ablation Assets/Ablation_Task_A_CNN_Model')
        ohe = joblib.load('Task A Assets/Task_A_one-hot_encoder.joblib')
        ohe_depth = 2
    elif task == 'task_b':
        loaded_model = keras.models.load_model('Ablation Assets/Ablation_Task_B_CNN_Model')
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
    print(classification_report(y_test_class, result_class))
    
