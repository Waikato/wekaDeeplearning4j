# Development

This document provides information that are specific to the development of Wekadeeplearning4j.

## Publish a new Release

See the [RELEASE.md](./RELEASE.md) documentation.

## Build the Weka Package

The `build.py` script handles building (and optionally installing) the Weka
package (calling Gradle underneath). This is also useful for local tests inside
the Weka GUI. Check out `build.py --help` for more information.


## Run JUnit Tests

Run the Gradle test target:
```bash
$ ./gradlew test
```

## Add/Remove Library Dependencies

Dependencies are managed by Gradle and specified in the `gradle.build` file in
the dependencies block:

```groovy
dependencies {
    ...
}
```

## Update CUDA Versions

If a new version of Deeplearning4j updates its supported CUDA version, the
following files need adjustments:

- `build.gradle`: update `valid_cuda_versions` variable
- `build.py`: update `CUDA_VERSIONS` variable
- `cuda-scripts/install-cuda-libs.sh`: update checks against valid cuda versions
- `cuda-scripts/install-cuda-libs.ps1`: update checks against valid cuda versions

## CUDA Library Installation Scripts

There are four scripts in `cuda-scripts`, namely:

- `install-cuda-libs.sh`
- `uninstall-cuda-libs.sh`
- `install-cuda-libs.ps1`
- `uninstall-cuda-libs.ps1`

The install scripts are supposed to download, extract and move all necessary
CUDA libraries (jar files) into the appropriate WekaDeeplearning4j package installation
directory of the host that is running the script, w.r.t. the host's operating
system. 

The uninstall scripts on the other hand will remove (delete) the CUDA libraries
from the local WekaDeeplearning4j installation. 

Linux and MacOS users should use the bash files while Windows users need the
powershell files.


## Update Java Docs

Java docs reside at https://waikato.github.io/wekaDeeplearning4j.
The documentation is automatically generated (extracted from class/method
documentation in the Java files) and pushed to the `gh-pages` branch via:

```bash
$ ./update-javadocs.sh
```

## Correctly Setting Classpath
If you run into a `The filename or extension is too long` error, make sure to set the `Shorten Command line`
parameter in your run configuration to `@argfiles`.

## Updating Zoo Models

Currently there are two types of zoo models included in WekaDeeplearning4j: DL4J Zoo and Keras Applications.
Not all of the models in the DL4J Zoo have pretrained weights available, however all Keras models do. 
The table below specifies the progress on each model type, and contains notes if it wasn't able to be
implemented in WekaDeeplearning4j. Note that only image classification models are included; object detection
is not supported in WekaDeeplearning4j.

The models typically have a dense layer as the final layer, with 1024/2048/some large number of inputs, and # 
of classes outputs (e.g. 1000 for Imagenet). This is then stripped off (with the connections still intact) 
and an OutputLayer attached with the sane number of inputs, and the number of output classes we want (e.g. only 10).
This works fine for most models, however, some don't reduce to a 2d Dense layer before the output layer, 
so we have to attach an intermediary pooling layer (`requiresPooling` flag) before attaching the output layer.


### Current State

| Framework | Model             | Pretrained Implemented | Weights Available         | Varieties                                | Notes                                                                                                                                                                  |
|-----------|-------------------|------------------------|---------------------------|------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DL4J      | AlexNet           | No                     | -                         | Standard                                 |                                                                                                                                                                        |
| DL4J      | Darknet19         | Yes                    | ImageNet                  | Standard (224x224 or 448x448 input size) | Attaches to 1000 class layer                                                                                                                                           |
| DL4J      | FaceNetNN4Small2  | No                     | -                         | Standard                                 |                                                                                                                                                                        |
| DL4J      | InceptionResNetV1 | No                     | -                         | Standard                                 |                                                                                                                                                                        |
| DL4J      | LeNet             | Yes                    | MNIST                     | Standard                                 |                                                                                                                                                                        |
| DL4J      | NASNet            | No                     | ImageNet, ImageNet Large  | Standard                                 | Bug in DL4J model builder code                                                                                                                                         |
| DL4J      | ResNet50          | Yes                    | ImageNet                  | Standard                                 |                                                                                                                                                                        |
| DL4J      | SimpleCNN         | No                     | -                         | Standard                                 | Bug in initialization code, removed from the package                                                                                                                                                                       |
| DL4J      | SqueezeNet        | Yes                    | ImageNet                  | Standard                                 | Attaches to 1000 class layer                                                                                                                                           |
| DL4J      | VGG               | Yes                    | ImageNet, VGGFace (16 only)| 16, 19                                  | CIFAR10 weights are in a legacy format so don't work.                                                                                                                  |
| DL4J      | XCeption          | Yes                    | ImageNet                  | Standard                                 | Attaches to 1000 class layer                                                                                                                                           |
| Keras     | DenseNet          | Yes                    | ImageNet                  | 121, 169, 201                            |                                                                                                                                                                        |
| Keras     | EfficientNet      | Yes                    | ImageNet                  | B0-B7                                    | Waiting on new DL4J release to make model functions available.                                                                                                         |
| Keras     | InceptionResNetV2 | No                     | ImageNet                  | Standard                                 | Requires writing custom lambda layers (TODO)                                                                                                                           |
| Keras     | InceptionV3       | Yes                    | ImageNet                  | Standard                                 |                                                                                                                                                                        |
| Keras     | MobileNet         | No                     | ImageNet                  | V1, V2                                   | Uses [relu layer](https://keras.io/layers/advanced-activations/) instead of [relu activation function](https://keras.io/activations/), which is not supported in DL4J. |
| Keras     | NASNet            | Yes                    | ImageNet                  | Mobile, Large                            |                                                                                                                                                                        |
| Keras     | ResNet            | Yes                    | ImageNet                  | 50, 50V2, 101, 101V2, 152, 152V2         |                                                                                                                                                                        |
| Keras     | VGG               | Yes                    | ImageNet                  | 16, 19                                   |                                                                                                                                                                        |
| Keras     | Xception          | Yes                    | ImageNet                  | Standard                                 |                                                                                                                                                                        |

### Adding new Zoo models

In a recent release of DL4J, importing Keras models via `.h5` files broke for some model types.

[Github Issue](https://github.com/eclipse/deeplearning4j/issues/8976)

To remedy this, Keras model loading is now done via the raw DL4J format.

All of these steps should be run from within the `weka/dl4j/scripts/keras_downloading` folder

- Set up python environment. Recommended to use Anaconda.
- Run the keras downloader: `python keras_download.py`. This downloads and saves Keras models as specified in `models.py`.
At this point you could load these `.h5` files directly, but as mentioned above, that method doesn't work in later versions of DL4J.
- Convert the `.h5` files into `.zip`. This is done by running the `KerasModelConverter` script. 
Provide it with the location of the h5 files e.g., 
```shell script
java KerasModelConverter src/main/weka/dl4j/scripts/output_h5 src/main/weka/dl4j/scripts/output_summary
```
- In the `dl4j_format` folder, you should now have `.zip` files for all models that were successfully converted.
Check the logs for information on models that couldn't be converted.

Check out the [Conversion README](src/main/java/weka/dl4j/scripts/keras_downloading/README.md) for more info.
Note that you only need to do these steps if attempting to release new versions of the models - this is unlikely to be needed
for the currently implemented models, as the pretrained weights don't change.

### Reverse Channels

Some models require the input image channels to be reversed (currently only `EfficientNet`) due to the way the weights
were saved and parsed. To set this for a model, simply add `setChannelsLast(true);` to the constructor. This change
will be propogated to any ImageInstanceIterators used with the model.


