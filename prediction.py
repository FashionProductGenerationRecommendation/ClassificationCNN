import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import save_img
from keras.models import load_model

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

model_file_h5 = "model-weights/Model.h5"
testFashionProductPath = 'fashion-dataset/images/1890.jpg'
save_classes = "model-classes/classes.npy"
classes = np.load(save_classes)
model = load_model(model_file_h5)
mostVisuallySimilarProductProperties = performPrediction(model, classes, testFashionProductPath)
print ("\nMost Visually Similar Product Properties")
print (mostVisuallySimilarProductProperties)