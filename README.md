# wekaDeeplearning4j

![Logo](docs/img/Weka_3_full.png)

DL4J wrapper for WEKA. Original code written by Mark Hall. This package currently introduces a new classifier,
`Dl4jMlpClassifier`, which allows arbitrary-depth MLPs to be built with a degree of flexibility (e.g. type of weight initialisation,
loss function, gradient descent algorithm, etc.).

The full documentation, giving installation instructions and getting started guides, is available [here](https://deeplearning.cms.waikato.ac.nz/).

![Weka Workbench GUI](docs/img/gui.png)

## Installation with Pre-Built Zip
The [latest release](https://github.com/Waikato/wekaDeeplearning4j/releases/latest) provides a pre-built zip file of the package that allow easy installation via commandline
```bash
java -cp weka.jar weka.core.WekaPackageManager \
     -install-package package.zip
```

or via the GUI package manager as described [here](http://weka.wikispaces.com/How+do+I+use+the+package+manager%3F#toc2).

### GPU Support

To add GPU support, [download](https://github.com/Waikato/wekaDeeplearning4j/releases/latest) and run the latest `install-cuda-libs.sh` for Linux/Macosx or `install-cuda-libs.ps1` for Windows. Make sure CUDA is installed on your system as explained [here](https://deeplearning.cms.waikato.ac.nz/install/#gpu).

The install script automatically downloads the libraries and copies them into your wekaDeeplearning4j package installation. If you want to download the library zip yourself, choose the appropriate combination of your platform and CUDA version from the [latest release](https://github.com/Waikato/wekaDeeplearning4j/releases/latest) and point the installation script to the file, e.g.:
```bash
./install-cuda.sh ~/Downloads/wekaDeeplearning4j-cuda-9.1-1.5.0-linux-x86_64.zip
```

## Usage

Example scipts are provided in the `weka-run-test-scripts` directory, e.g.:
```bash
$ java -cp ${WEKA_HOME}/weka.jar weka.Run \
       .Dl4jMlpClassifier \
       -S 1 \
       -layer "weka.dl4j.layers.DenseLayer -nOut 32 -activation \"weka.dl4j.activations.ActivationReLU \" " \
       -layer "weka.dl4j.layers.OutputLayer -activation \"weka.dl4j.activations.ActivationSoftmax \" " \
       -numEpochs 10 \
       -t ../datasets/nominal/iris.arff
```

## Documentation
The full documentation, giving installation instructions and getting started guides, is available at [https://deeplearning.cms.waikato.ac.nz/](https://deeplearning.cms.waikato.ac.nz/).

The java documentation can be found [here](https://waikato.github.io/wekaDeeplearning4j/).

## Contributing

If you want to contribute to the project, check out the [contributing guide](https://github.com/Waikato/wekaDeeplearning4j/blob/master/CONTRIBUTING.md).

## Misc.
Original code by Mark Hall
