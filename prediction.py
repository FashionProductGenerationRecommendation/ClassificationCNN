# TEAM NAME : KRAB
# TEAM MEMBERS : Annuj Jain, Bharat Goel, Keshav Aditya Rajupet Premkumar, Rutvij Mehta
# 1.  References Used For Convoluted Neural Network (CNN):
#     https://www.analyticsvidhya.com/blog/2019/04/build-first-multi-label-image-classification-model-python/
# 2.  GENERAL DESCRIPTION OF CODE :
#     We use the CNN to find the most visually similar product in the datset given a test image. 
#     The Features X : Are all the Fashion Product Images converted to array format 
#     The Target Y : Are the the product description like color, usage, category, gender etc
#     Then for a given test image, the cnn will predict all its descriptions ie. Its color, gender, usage etc
#     [This file is used predict fashion product image description using the trained model]
# 3.  DATA FRAMEWORK USED : Tensorflow (Keras)
# 4.  CONCEPT USED : Deep Learning (Convoluted Neural Netwrok)
# 5.  SYSTEM TO RUN CODE ON : 128 GB RAM | 16 core CPU | 100 GB persistent memory VM on Google Cloud Compute 

import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import save_img
from keras.models import load_model

# Use the trained CNN Model to predict the descriptions of the Test Image
def performPrediction(model, classes, testFashionProductPath):
    testFashionProductImage = load_img(testFashionProductPath)
    testFashionProductImageArray = img_to_array(testFashionProductImage)
    testFashionProductImageArray = tf.image.resize(testFashionProductImageArray, [50, 50], method='bilinear', preserve_aspect_ratio=True, antialias=True, name=None)
    testFashionProductImageArray = testFashionProductImageArray / 255
    testFashionProductImageArrayReshapped = tf.keras.backend.reshape(testFashionProductImageArray, shape=(1,50,38,3))
    proba = model.predict(testFashionProductImageArrayReshapped, steps=1)
    proba = proba.flatten()
    lengthClasses = len(classes)
    output = {}
    output["baseColor"] = []
    output["gender"] = []
    output["usage"] = []
    output["masterCategory"] = []
    output["subCategory"] = []
    output["articleType"] = []
    output["season"] = []
    for i in range(0,lengthClasses):
        data = {}
        data["label"] = classes[i]
        data["sigmoid"] = proba[i]
        if "baseColor" in data["label"]:
            output["baseColor"].append(data)
        elif "gender" in data["label"]:
            output["gender"].append(data)
        elif "usage" in data["label"]:
            output["usage"].append(data)
        elif "masterCategory" in data["label"]:
            output["masterCategory"].append(data)
        elif "subCategory" in data["label"]:
            output["subCategory"].append(data)
        elif "articleType" in data["label"]:
            output["articleType"].append(data)
        elif "season" in data["label"]:
            output["season"].append(data)
        else:
            print ("Something Wrong", data)
    prediction = {}
    for key, value in output.items():
        topResultInCategory = sorted(value, key=lambda x: x["sigmoid"])[-1] 
        selectedCategoryOption = (topResultInCategory["label"].lower()).replace(key.lower()+"_","")
        prediction[key] = {}
        prediction[key]["label"] = selectedCategoryOption
        prediction[key]["sigmoid"] = topResultInCategory["sigmoid"]
    print ("\nCNN Result")
    print (output)
    return prediction

# Main
model_file_h5 = "model-weights/Model.h5"
testFashionProductPath = 'fashion-dataset/images/1890.jpg'
save_classes = "model-classes/classes.npy"
classes = np.load(save_classes)
model = load_model(model_file_h5)
mostVisuallySimilarProductProperties = performPrediction(model, classes, testFashionProductPath)
print ("\nMost Visually Similar Product Properties")
print (mostVisuallySimilarProductProperties)