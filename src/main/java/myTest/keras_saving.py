# # from keras.applications.resnet import ResNet152V2 as model_create
# import keras_applications
# from keras.models import Sequential
from os import path
# import keras


save_path = "."
# model_name = 'effNetB0'
# keras_model = model_create()
# print(keras_model.summary())
# # vgg_model.summary()
# # model = Sequential()

# # for layer in vgg_model.layers:
# #     model.add(layer)

# print("Finished loading in")
# # print(model.input)
# # print(model.summary())

# keras_model.save(path.join(save_path, model_name + ".h5"))
# # with open(path.join(save_path, this_model + ".json"), mode='w') as f:
#     # f.write(vgg_model.to_json())

# NOTE: org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException: Model configuration attribute missing from

import efficientnet.keras as efn 

model = efn.EfficientNetB0(weights='imagenet')
model_name = 'efficientnet-b0'

# model.save(path.join(save_path, model_name + ".h5"))
print(model.summary())