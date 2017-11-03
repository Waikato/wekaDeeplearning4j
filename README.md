# wekaDeeplearning4j
DL4J wrapper for WEKA. Original code written by Mark Hall. This package currently introduces a new classifier,
`Dl4jMlpClassifier`, which allows arbitrary-depth MLPs to be built with a degree of flexibility (e.g. type of weight initialisation,
loss function, gradient descent algorithm, etc.).

The full documentation, giving installation instructions and getting started guides, is available [here](https://waikato.github.io/wekaDeeplearning4j/).

![Weka workbench GUI](docs/img/gui.png)

## Installation
### Use pre-built zips
The [latest release](https://github.com/Waikato/wekaDeeplearning4j/releases/latest) provide pre-built zip files of the packages that allow easy installation via commandline
```bash
java -cp weka.jar weka.core.WekaPackageManager \
     -install-package package.zip
```

or via the GUI package manager as described [here](http://weka.wikispaces.com/How+do+I+use+the+package+manager%3F#toc2).

### Build locally
Simply run the `build.sh` script. This assumes:
* You have Ant and Maven installed.
* WEKA's `weka.jar` file resides somewhere in your Java classpath. The latest and greatest WEKA installation is highly recommended; you
  can get the .jar of the nightly snapshot [here](http://www.cs.waikato.ac.nz/~ml/weka/snapshots/developer-branch.zip).

```
Usage: build.sh

Optional arguments:
   -v/--verbose            Enable verbose mode
   -i/--install-packages   Install selected packages
   -b/--backend            Select specific backend 
                           Available: ( CPU GPU )
   -c/--clean              Clean up build-environment
   -h/--help               Show this message
```

## Usage

An example script is provided that can be run on the Iris dataset in the `scripts` directory.

```bash
java -Xmx5g -cp ${WEKA_HOME}/weka.jar weka.Run \
     .Dl4jMlpClassifier -S 1 \
     -normalization "Standardize training data" \
     -layer "weka.dl4j.layers.OutputLayer -activation \"weka.dl4j.activations.ActivationSoftmax \" -lossFn \"weka.dl4j.lossfunctions.LossMCXENT \" " \
     -config "weka.dl4j.NeuralNetConfiguration -updater ADAM" \
     -numEpochs 10 \
     -t datasets/nominal/iris.arff

```

## Documentation

The java-doc can be found [here](https://waikato.github.io/wekaDeeplearning4j/doc/).

## Design philosophy

DL4J is not primarily intended for research purposes -- rather more commercial and convention endeavours -- and so for more research-oriented
tasks, a library such as Theano should be used in conjunction with the WekaPyScript package which allows WEKA classifiers to be prototyped in
Python.

## Misc
Original code by Mark Hall
