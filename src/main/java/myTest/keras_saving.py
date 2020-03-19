from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras import Sequential
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from os import path
import keras

save_path = "/home/rhys/Documents/git/wekaDeeplearning4j/src/main/resources/"

this_model = "mobilenetv2"
vgg_model = keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet')
# # vgg_model.summary()
# model = Sequential()
# for layer in vgg_model.layers:
#     model.add(layer)

# model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['accuracy'])

vgg_model.save(path.join(save_path, this_model + ".h5"))
with open(path.join(save_path, this_model + ".json"), mode='w') as f:
    f.write(vgg_model.to_json())