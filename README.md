# wekaDeeplearning4j

![Logo](docs/img/Weka_3_full.png)

DL4J wrapper for WEKA. Original code written by Mark Hall. This package currently introduces a new classifier,
`Dl4jMlpClassifier`, which allows arbitrary-depth MLPs to be built with a degree of flexibility (e.g. type of weight initialisation,
loss function, gradient descent algorithm, etc.).

The full documentation, giving installation instructions and getting started guides, is available [here](https://deeplearning.cms.waikato.ac.nz/).

![Weka Workbench GUI](docs/img/gui.png)

## Installation with Pre-Built Zips 
The [latest release](https://github.com/Waikato/wekaDeeplearning4j/releases/latest) provide pre-built zip files of the packages that allow easy installation via commandline
```bash
java -cp weka.jar weka.core.WekaPackageManager \
     -install-package package.zip
```

or via the GUI package manager as described [here](http://weka.wikispaces.com/How+do+I+use+the+package+manager%3F#toc2).

## Usage

An example script is provided that can be run on the Iris dataset in the `scripts` directory.
```bash
$ java -cp $WEKA_HOME/weka.jar weka.Run \
		.Dl4jMlpClassifier \
		-S 1 \
		-layer "weka.dl4j.layers.OutputLayer \
		        -activation weka.dl4j.activations.ActivationSoftmax \
		        -lossFn weka.dl4j.lossfunctions.LossMCXENT" \
		-config "weka.dl4j.NeuralNetConfiguration \
		        -updater weka.dl4j.updater.Adam" \
		-numEpochs 10 \
		-t datasets/nominal/iris.arff \
		-split-percentage 66
```

## Documentation

The java documentation can be found [here](https://waikato.github.io/wekaDeeplearning4j/doc/).

## Contributions

Contributions are welcome and an easy way to get started is to file an issue. Make sure to be as descriptive about your problem as possible. Try to explain what you have tried, what you expected and what the actual outcome was. Give additional information about your java and weka version, as well as platform specific details that could be relevant. 

If you are going to contribute to the codebase, you should fork this repository, create a separate branch on which you commit your changes and file a pull request. A well explained how-to is described [here](https://gist.github.com/Chaser324/ce0505fbed06b947d962).

### Java Code Style
This package mostly follows the official [Google Java Style Guide](https://google.github.io/styleguide/javaguide.html).

### Build Locally
Simply run the `build.sh` script. This assumes:
* Bash
* GNU grep
* GNU sed
* Ant
* Maven
* Weka

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

## Misc.
Original code by Mark Hall
