import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import glob
import json
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import save_img

fashionProducts = []
X = []
Y = []
maximumNumberOfFashionProducts = 30
count = 1
jsonPath = "fashion-dataset/styles/"
imagesPath = 'fashion-dataset/images/*.jpg'
for filename in glob.glob(imagesPath):
    item = {}
    fashionProductId = filename.strip('fashion-dataset/images/').strip('.jpg')
    fashionProductImage = load_img(filename)
    fashionProductImageArray = img_to_array(fashionProductImage)
    fashionProductImageArray = tf.image.resize(fashionProductImageArray, [50, 50], method='bilinear', preserve_aspect_ratio=True, antialias=True, name=None)
    fashionProductImageArray = fashionProductImageArray / 255
    # save_img(str(count)+'.jpg', fashionProductImageArray)
    f = open(jsonPath + fashionProductId + '.json',)
    fashionProductDetails = json.load(f)
    item["id"] = fashionProductId
    item["baseColor"] = fashionProductDetails["data"]["baseColour"]
    item["gender"] = fashionProductDetails["data"]["gender"]
    item["usage"] = fashionProductDetails["data"]["usage"]
    item["fashionImage"] = fashionProductImageArray
    if item["id"] and item["baseColor"] and item["gender"] and item["usage"] :
        fashionProducts.append([int(item["id"]), item["baseColor"], item["gender"], item["usage"]])
        X.append(fashionProductImageArray)
    count += 1
    if count > maximumNumberOfFashionProducts:
        break
X = np.array(X)
df = pd.DataFrame.from_records(fashionProducts, columns=['id', 'baseColor', 'gender', 'usage'])
# df = df.join(pd.get_dummies(df['baseColor']))
df = pd.get_dummies(df, prefix=['baseColor', 'gender', 'usage'])
print (list(df.columns.values))
Y = np.array(df.drop(['id'], axis=1))
# print (df)
# print (X)
# print (df)
print (X.shape)
print (Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

