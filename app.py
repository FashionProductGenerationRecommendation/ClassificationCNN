# TEAM NAME : KRAB
# TEAM MEMBERS : Annuj Jain, Bharat Goel, Keshav Aditya Rajupet Premkumar, Rutvij Mehta
# 1.  References Used For Convoluted Neural Network (CNN):
#     https://www.analyticsvidhya.com/blog/2019/04/build-first-multi-label-image-classification-model-python/
# 2.  GENERAL DESCRIPTION OF CODE :
#     We use the CNN to find the most visually similar product in the datset given a test image. 
#     The Features X : Are all the Fashion Product Images converted to array format 
#     The Target Y : Are the the product description like color, usage, category, gender etc
#     Then for a given test image, the cnn will predict all its descriptions ie. Its color, gender, usage etc
#     [This file is used to tain the CNN]
# 3.  DATA FRAMEWORK USED : Tensorflow (Keras)
# 4.  CONCEPT USED : Deep Learning (Convoluted Neural Netwrok)
# 5.  SYSTEM TO RUN CODE ON : 128 GB RAM | 16 core CPU | 100 GB persistent memory VM on Google Cloud Compute 

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import glob
import json
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import save_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import re

# Process the Fashion Image Data and Image Description to create X and Y
def createFeaturesAndTargets(maximumNumberOfFashionProducts = 30):
    misMatchDimensions = []
    fashionProducts = []
    X = []
    Y = []
    count = 1
    imagesPath = 'fashion-dataset/images/*.jpg'
    stylesDataset = pd.read_csv("fashion-dataset/styles.csv", encoding='utf-8', error_bad_lines=False)
    stylesDataset = stylesDataset.drop_duplicates(subset='id', keep="last")
    for filename in glob.glob(imagesPath):
        fashionProductId = filename.strip('fashion-dataset/images/').strip('.jpg')
        fashionProductImage = load_img(filename)
        fashionProductImageArray = img_to_array(fashionProductImage)
        fashionProductImageArray = tf.image.resize(fashionProductImageArray, [50, 50], method='bilinear', preserve_aspect_ratio=True, antialias=True, name=None)
        fashionProductImageArray = fashionProductImageArray / 255
        if (fashionProductImageArray.shape != (50, 38, 3)):
            misMatchDimensions.append(fashionProductId)
            count += 1
            continue
        item = {}
        fashionProductId = int(fashionProductId)
        selectedProductRow = stylesDataset[stylesDataset['id'] == fashionProductId]
        if selectedProductRow.empty:
            continue
        item["id"] = fashionProductId
        item["baseColor"] = selectedProductRow['baseColour'].values[0]
        item["gender"] = selectedProductRow['gender'].values[0]
        item["usage"] = selectedProductRow['usage'].values[0]
        item["masterCategory"] = selectedProductRow['masterCategory'].values[0]
        item["subCategory"] = selectedProductRow['subCategory'].values[0]
        item["articleType"] = selectedProductRow['articleType'].values[0]
        item["season"] = selectedProductRow['season'].values[0]
        item["fashionImage"] = fashionProductImageArray
        if item["id"] and item["baseColor"] and item["gender"] and item["usage"] and item["masterCategory"] and item["subCategory"] and item["articleType"] and item["season"]:
            fashionProducts.append([item["id"], item["baseColor"], item["gender"], item["usage"], item["masterCategory"], item["subCategory"], item["articleType"], item["season"]])
            X.append(item["fashionImage"])
        count += 1
        if count > maximumNumberOfFashionProducts:
            break
    X = np.array(X)
    df = pd.DataFrame.from_records(fashionProducts, columns=['id', 'baseColor', 'gender', 'usage', 'masterCategory', 'subCategory', 'articleType', 'season'])
    df = pd.get_dummies(df, prefix=['baseColor', 'gender', 'usage', 'masterCategory', 'subCategory', 'articleType', 'season'])
    Y = np.array(df.drop(['id'], axis=1))
    classes = np.array(list(df.columns.values)[1:])
    print ("\n Total Number of Fashion Product Images")
    print (count)
    print ("\nTarget Headings")
    print (classes)
    print ("\nFeatures Shape")
    print (X.shape)
    print ("\nTarget Shape")
    print (Y.shape)
    print ("\n Items Excluded")
    print (misMatchDimensions)
    return X, Y, classes

# CNN Architecture Description
def createCnnModel(numberOfTargetColumns):
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(50,38,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu", padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numberOfTargetColumns, activation='sigmoid'))
    model.summary()
    return model

# Compile and Fit the Model
def runCNN(model, X_train, X_test, y_train, y_test):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=128)
    return model

# Main
maximumNumberOfFashionProducts = 60000
model_file_h5 = "model-weights/Model.h5"
save_classes = "model-classes/classes.npy"
X, Y, classes = createFeaturesAndTargets(maximumNumberOfFashionProducts)
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.3)
numberOfTargetColumns = len(classes)
model = createCnnModel(numberOfTargetColumns)
model = runCNN(model, X_train, X_test, y_train, y_test)
model.save(model_file_h5)
np.save(save_classes, classes)
print("Saved model to disk")