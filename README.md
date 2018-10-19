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
./install-cuda.sh ~/Downloads/wekaDeeplearning4j-cuda-9.2-1.5.9-linux-x86_64.zip
```

## Usage
As most of Weka, WekaDeeplearning4j's functionality is accessible in three ways:

- Using the Weka workbench GUI
- Programming with Weka in Java
- Via the commandline interface

All three ways are explained in the [getting-started](https://deeplearning.cms.waikato.ac.nz/user-guide/getting-started/) documentation. 

Example commandline scripts are provided in the `weka-run-test-scripts` directory, e.g. a simple network with one dense layer of 32 neurons and one output layer, classifying the iris dataset, would look like the following:
```bash
$ java -cp ${WEKA_HOME}/weka.jar weka.Run \
       .Dl4jMlpClassifier \
       -layer "weka.dl4j.layers.DenseLayer -nOut 32 -activation \"weka.dl4j.activations.ActivationReLU \" " \
       -layer "weka.dl4j.layers.OutputLayer -activation \"weka.dl4j.activations.ActivationSoftmax \" " \
       -numEpochs 30 \
       -t datasets/nominal/iris.arff
```

which results in:

```
=== Stratified cross-validation ===

Correctly Classified Instances         141               94      %
Incorrectly Classified Instances         9                6      %
Kappa statistic                          0.91  
Mean absolute error                      0.0842
Root mean squared error                  0.1912
Relative absolute error                 18.9359 %
Root relative squared error             40.5586 %
Total Number of Instances              150     


=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 1.000    0.000    1.000      1.000    1.000      1.000    1.000     1.000     Iris-setosa
                 0.880    0.030    0.936      0.880    0.907      0.864    0.978     0.928     Iris-versicolor
                 0.940    0.060    0.887      0.940    0.913      0.868    0.979     0.972     Iris-virginica
Weighted Avg.    0.940    0.030    0.941      0.940    0.940      0.911    0.986     0.967     


=== Confusion Matrix ===

  a  b  c   <-- classified as
 50  0  0 |  a = Iris-setosa
  0 44  6 |  b = Iris-versicolor
  0  3 47 |  c = Iris-virginica
```

The same Setup can be achieved in Java with the following code:
```java
// Setup layers
DenseLayer dense = new DenseLayer();
dense.setNOut(32);
OutputLayer out = new OutputLayer();
        
// Setup MLP
Dl4jMlpClassifier mlp = new Dl4jMlpClassifier();
mlp.setLayers(dense, out);
        
// Build model
mlp.buildClassifier(loadIris());
```

## Documentation
The full documentation, giving installation instructions and getting started guides, is available at [https://deeplearning.cms.waikato.ac.nz/](https://deeplearning.cms.waikato.ac.nz/).

The java documentation can be found [here](https://waikato.github.io/wekaDeeplearning4j/).

## Contributing

If you want to contribute to the project, check out the [contributing guide](https://github.com/Waikato/wekaDeeplearning4j/blob/master/CONTRIBUTING.md).

## Misc.
Original code by Mark Hall
