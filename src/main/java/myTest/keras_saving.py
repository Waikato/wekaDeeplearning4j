from os import path
import efficientnet.keras as efn
import keras
import efficientnet
from multiprocessing.pool import ThreadPool
save_path = "."

models = [
    (keras.applications.xception.Xception, 'Xception'),
    (keras.applications.vgg16.VGG16, 'VGG16'),
    (keras.applications.vgg19.VGG19, 'VGG19'),
    (keras.applications.resnet.ResNet50, 'ResNet50'),
    (keras.applications.resnet.ResNet101, 'ResNet101'),
    (keras.applications.resnet.ResNet152, 'ResNet152'),
    (keras.applications.resnet_v2.ResNet50V2, 'ResNet50V2'),
    (keras.applications.resnet_v2.ResNet101V2, 'ResNet101V2'),
    (keras.applications.resnet_v2.ResNet152V2, 'ResNet152V2'),
    (keras.applications.inception_v3.InceptionV3, 'InceptionV3'),
    (keras.applications.inception_resnet_v2.InceptionResNetV2, 'InceptionResNetV2'),
    (keras.applications.mobilenet.MobileNet, 'MobileNet'),
    (keras.applications.mobilenet_v2.MobileNetV2, 'MobileNetV2'),
    (keras.applications.densenet.DenseNet121, 'DenseNet121'),
    (keras.applications.densenet.DenseNet169, 'DenseNet169'),
    (keras.applications.densenet.DenseNet201, 'DenseNet201'),
    (keras.applications.nasnet.NASNetLarge, 'NASNetLarge'),
    (keras.applications.nasnet.NASNetMobile, 'NASNetMobile'),
    (efn.EfficientNetB0, 'EfficientNetB0'),
    (efn.EfficientNetB1, 'EfficientNetB1'),
    (efn.EfficientNetB2, 'EfficientNetB2'),
    (efn.EfficientNetB3, 'EfficientNetB3'),
    (efn.EfficientNetB4, 'EfficientNetB4'),
    (efn.EfficientNetB5, 'EfficientNetB5'),
    (efn.EfficientNetB6, 'EfficientNetB6'),
    (efn.EfficientNetB7, 'EfficientNetB7'),
]

def download_and_save_model(model_def):
    model_fn = model_def[0]
    model_name = model_def[1]

    # Download the model
    keras_model = model_fn()

    # Save the summary (for debugging)
    with open(path.join(save_path, model_name + "_summary.txt"), mode='w') as fh:
        keras_model.summary(print_fn=lambda x: fh.write(x + '\n'))
    
    # Save the model config
    with open(path.join(save_path, model_name + ".json"), mode='w') as fh:
        fh.write(keras_model.to_json())

    # # Save the model weights
    # keras_model.save(path.join(save_path, model_name + ".h5"))

    return model_name

results = ThreadPool(8).imap_unordered(download_and_save_model, models)

for name in results:
    print("\n\n", name, "saved\n\n")

# download_and_save_model((keras.applications.resnet.ResNet50, 'ResNet50'))