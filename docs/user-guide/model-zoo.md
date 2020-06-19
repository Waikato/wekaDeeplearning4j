# Model Zoo

WekaDeeplearning4J contains a wide range of popular architectures, ready to use either for training or as feature extractors.
The table below outlines the different models included, whether pretrained weights are available, the types of pretrained weights,
and the model variations (if any). WekaDeeplearning4j merges the model zoo of Deeplearning4j *and* Keras.
Values in **bold** are the defaults.

| Framework | Model             | Pretrained Implemented | Weights Available                  | Varieties                            |
|-----------|-------------------|------------------------|------------------------------------|--------------------------------------|
| DL4J      | AlexNet           | No                     | -                                  | -                                    |
| DL4J      | Darknet19         | Yes                    | **ImageNet**                       | **224x224** or 448x448 input size    |
| DL4J      | FaceNetNN4Small2  | No                     | -                                  | -                                    |
| DL4J      | InceptionResNetV1 | No                     | -                                  | -                                    |
| DL4J      | LeNet             | Yes                    | **MNIST**                          | -                                    |
| DL4J      | ResNet50          | Yes                    | **ImageNet**                       | -                                    |
| DL4J      | SqueezeNet        | Yes                    | **ImageNet**                       | -                                    |
| DL4J      | VGG               | Yes                    | **ImageNet**, VGGFace (VGG16 only) | **16**, 19                           |
| DL4J      | XCeption          | Yes                    | **ImageNet**                       | -                                    |
| Keras     | DenseNet          | Yes                    | **ImageNet**                       | **121**, 169, 201                    |
| Keras     | EfficientNet      | Yes                    | **ImageNet**                       | **B0**-B7                            |
| Keras     | InceptionV3       | Yes                    | **ImageNet**                       | -                                    |
| Keras     | NASNet            | Yes                    | **ImageNet**                       | **Mobile**, Large                    |
| Keras     | ResNet            | Yes                    | **ImageNet**                       | **50**, 50V2, 101, 101V2, 152, 152V2 |
| Keras     | VGG               | Yes                    | **ImageNet**                       | **16**, 19                           |
| Keras     | Xception          | Yes                    | **ImageNet**                       | -                                    |

The EfficientNet family of models will be added soon.

To set a predefined model, e.g. ResNet50, from the model zoo in the GUI is straightforward via the corresponding pop-up menu. 
To set a predefined model from the command-line or via the API, it is necessary to add the 
`-zooModel ".Dl4JResNet50"` option via commandline, or call the `setZooModel(new ResNet50())` on the `Dl4jMlpClassifier` object.

Model names from Keras are prepended with `Keras`, i.e., `KerasResNet`, and similarly for Deeplearning4j models (e.g., `DL4JDarknet19`).
In addition, some models support different variations. Again, it is straightforward to do this via the GUI. 
To do via command line you must add the `-variation` argument e.g.:

```shell
...
-ZooModel ".KerasResNet -variation RESNET152V2" 
...
```

If using the Java API, these can be set via `.setVariation()` e.g.:

```java
KerasResNet kerasResNet = new KerasResNet();
kerasResNet.setVariation(ResNet.VARIATION.RESNET152V2);
```

View the [featurizing tutorial](../examples/featurize-mnist.md) and the [finetuning tutorial](../examples/classifying-your-own.md)
for usage examples with the model zoo.

### Model Summaries

The following summaries are generated from the pretrained zoo models included in WekaDeeplearning4j. 
These may be useful as a reference for trying different feature extraction layers, or simply to
investigate famous model architectures.

#### DL4J
* [DL4JDarkNet19](../model-zoo/dl4j/DL4JDarkNet19.md)
* [DL4JLeNet](../model-zoo/dl4j/DL4JLeNet.md)
* [DL4JResNet50](../model-zoo/dl4j/DL4JResNet50.md)
* [DL4JSqueezeNet](../model-zoo/dl4j/DL4JSqueezeNet.md)
* [DL4JVGG16](../model-zoo/dl4j/DL4JVGG16.md)
* [DL4JVGG19](../model-zoo/dl4j/DL4JVGG19.md)
* [DL4JXception](../model-zoo/dl4j/DL4JXception.md)

#### Keras
* DenseNet
    * [KerasDenseNet121](../model-zoo/keras/KerasDenseNet121.md)
    * [KerasDenseNet169](../model-zoo/keras/KerasDenseNet169.md)
    * [KerasDenseNet201](../model-zoo/keras/KerasDenseNet201.md)
* EfficientNet
    * [KerasEfficientNetB0](../model-zoo/keras/KerasEfficientNetB0.md)
    * [KerasEfficientNetB1](../model-zoo/keras/KerasEfficientNetB1.md)
    * [KerasEfficientNetB2](../model-zoo/keras/KerasEfficientNetB2.md)
    * [KerasEfficientNetB3](../model-zoo/keras/KerasEfficientNetB3.md)
    * [KerasEfficientNetB4](../model-zoo/keras/KerasEfficientNetB4.md)
    * [KerasEfficientNetB5](../model-zoo/keras/KerasEfficientNetB5.md)
    * [KerasEfficientNetB6](../model-zoo/keras/KerasEfficientNetB6.md)
    * [KerasEfficientNetB7](../model-zoo/keras/KerasEfficientNetB7.md)
* InceptionV3
    * [KerasInceptionV3](../model-zoo/keras/KerasInceptionV3.md)
* NASNet
    * [KerasNASNetMobile](../model-zoo/keras/KerasNASNetMobile.md)
    * [KerasNASNetLarge](../model-zoo/keras/KerasNASNetLarge.md)
* ResNet
    * [KerasResNet50](../model-zoo/keras/KerasResNet50.md)
    * [KerasResNet50v2](../model-zoo/keras/KerasResNet50V2.md)
    * [KerasResNet101](../model-zoo/keras/KerasResNet101.md)
    * [KerasResNet101V2](../model-zoo/keras/KerasResNet101V2.md)
    * [KerasResNet152](../model-zoo/keras/KerasResNet152.md)
    * [KerasResNet152V2](../model-zoo/keras/KerasResNet152V2.md)
* VGG
    * [KerasVGG16](../model-zoo/keras/KerasVGG16.md)
    * [KerasVGG19](../model-zoo/keras/KerasVGG19.md)
* Xception
    * [KerasXception](../model-zoo/keras/KerasXception.md)

