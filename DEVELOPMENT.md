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

## Updating Zoo Models

Currently there are two types of zoo models included in WekaDeeplearning4j: DL4J Zoo and Keras Applications.
Not all of the models in the DL4J Zoo have pretrained weights available, however all Keras models do. 
The table below specifies the progress on each model type, and contains notes if it wasn't able to be
implemented in WekaDeeplearning4j. Note that only image classification models are included; object detection
is not supported in WekaDeeplearning4j.

The models typically have a dense layer as the final layer, with 1024/2048/some large number of inputs, and # 
of classes outputs (e.g. 1000 for Imagenet). This is then stripped off (with the connections still intact) 
and an OutputLayer attached with the sane number of inputs, and the number of output classes we want (e.g. only 10).
This works fine for most models, however, some have a more complex final layer setup.

### Current State

| Framework | Model             | Pretrained Implemented | Notes                                                                                          |
|-----------|-------------------|------------------------|------------------------------------------------------------------------------------------------|
| DL4J      | AlexNet           | No                     | No weights available                                                                           |
| DL4J      | Darknet19         | Yes                    | Attaches to 1000 class layer, can either be 224x224 or 448x448                                 |
| DL4J      | FaceNetNN4Small2  | No                     | No weights available                                                                           |
| DL4J      | InceptionResNetV1 | No                     | No weights available                                                                           |
| DL4J      | LeNet             | Yes                    | Only MNIST weights available                                                                   |
| DL4J      | NASNet            | Yes                    | Imagenet and Imagenet large available,  bug in DL4J model builder code                         |
| DL4J      | ResNet50          | Yes                    |                                                                                                |
| DL4J      | SimpleCNN         | No                     | No weights available                                                                           |
| DL4J      | SqueezeNet        | Yes                    | Attaches to 1000 class layer                                                                   |
| DL4J      | VGG16             | Yes                    | VGGFace, and ImageNet weights available. CIFAR10 weights are in a legacy format so don't work. |
| DL4J      | VGG19             | Yes                    |                                                                                                |
| DL4J      | XCeption          | Yes                    | Attaches to 1000 class layer                                                                   |
| Keras     | DenseNet          | Yes                    | 121, 169, 201 variants available                                                               |
| Keras     | EfficientNet      | No                     | B0-B7 available, waiting on new DL4J release to make model functions available.                |
| Keras     | InceptionResNetV2 | No                     | Weights available, but requires writing custom lambda layers (TODO)                            |
| Keras     | EfficientNetB0    | Yes                    |                                                                                                |
| Keras     |                   |                        |                                                                                                |


- Add a new class in /keras
- Add links to the model .h5 file and the relvant checksum in ...