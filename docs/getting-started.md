# Usage
If you are new to Weka, you should probably first start reading the [Weka primer](https://weka.wikispaces.com/Primer) as a basic introduction.

As most of Weka, the WekaDeeplearning4j's functionality is accessible in three ways:

- Via the commandline interface
- Programming with Weka in Java
- Using the Weka workbench GUI

All three ways are explained in the following. The main classifier exposed by this package is named `Dl4jMlpClassifier`.
Simple examples are given in the examples section for the [Iris dataset](examples/classifying-iris) and the [MNIST dataset](examples/classifying-mnist).

## Commandline Interface
A first look for the available commandline options of the `Dl4jMlpClassifier` is shown with
```bash
$ java -cp weka.jar weka.Run .Dl4jMlpClassifier -h
```
Below the general options, the specific ones are listed:
```
Options specific to weka.classifiers.functions.Dl4jMlpClassifier:

-S <num>
	Random number seed.
	(default 1)
-logFile <string>
	The name of the log file to write loss information to (default = no log file).
-layer <string>
	The specification of a layer. This option can be used multiple times.
-numEpochs <int>
	The number of epochs to perform.
-iterator <string>
	The dataset iterator to use.
-config <string>
	The neural network configuration to use.
-normalization <int>
	The type of normalization to perform.
-queueSize <int>
	The queue size for asynchronous data transfer (default: 0, synchronous transfer).
-output-debug-info
	If set, classifier is run in debug mode and
	may output additional info to the console
-do-not-check-capabilities
	If set, classifier capabilities are not checked before classifier is built
	(use with caution).
-num-decimal-places
	The number of decimal places for the output of numbers in the model (default 2).
-batch-size
	The desired batch size for batch prediction  (default 100).
```

The most interesting option may be the `-layer` specification. This option can be used multiple times and defines the architecture of the network layer-wise. 

```bash
$ java -cp weka.jar weka.Run \
       .Dl4jMlpClassifier \
       -layer "weka.dl4j.layers.DenseLayer \
              -activation weka.dl4j.activations.ActivationReLU \
              -nOut 10" \
       -layer "weka.dl4j.layers.OutputLayer \
              -activation weka.dl4j.activations.ActivationSoftmax \
              -lossFn weka.dl4j.lossfunctions.LossMCXENT" 
```
The above setup builds a network with one hidden layer, having 10 output units using the ReLU activation function, followed by an output layer with the softmax activation function, using a multi-class cross-entropy loss function (MCXENT) as optimization objective.

Another important option is the neural network configuration `-conf` in which you can setup hyperparameters for the network. Available options can be found in the [Java documentation](https://waikato.github.io/wekaDeeplearning4j/doc/weka/dl4j/NeuralNetConfiguration.html) (the field `commandLineParamSynopsis` indicates the commandline parameter name for each available method).


## Java
The Java API is a straight forward wrapper for the official DeepLearning4j API. Using the `Dl4jMlPClassifier` your code should usually start with
```java
// Create a new Multi-Layer-Perceptron classifier
Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
```

The networks architecture can be set up by creating each layer step by step:
```java
DenseLayer denseLayer = new DenseLayer();
denseLayer.setNOut(10);
denseLayer.setActivationFn(new ActivationReLU());
denseLayer.setWeightInit(WeightInit.XAVIER);

// Define the output layer
OutputLayer outputLayer = new OutputLayer();
outputLayer.setActivationFn(new ActivationSoftmax());
outputLayer.setUpdater(Updater.SGD);
outputLayer.setLearningRate(0.01);
outputLayer.setBiasLearningRate(0.01);
```

Further configuration can be done by setting a `NeuralNetConfiguration`
```java
NeuralNetConfiguration nnc = new NeuralNetConfiguration();
nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
clf.setNeuralNetConfiguration(nnc);
```

Finally the layers are set with
```java
// Add the layers to the classifier
clf.setLayers(new Layer[]{denseLayer, outputLayer});
```

## GUI
A tutorial on how to use the GUI is coming soon.

<!--- //TODO: uncomment as soon as ModelZoo is working again
# Model Zoo
WekaDeeplearning4j adapts the modelzoo of Deeplearning4j. That means it is possible to load predefined architectures as neural network and train it on a new dataset. Currently implemented architectures are:

- AlexNet
- LeNet
- SimpleCNN
- VGG16
- VGG19

This set of models will be extended over the time.

To set a predefined model, e.g. LeNet, from the modelzoo, it is necessary to add the `-zooModel "weka.dl4j.zoo.LeNet"` option via commandline, or call the `setZooModel(new LeNet())` on the `Dl4jMlpClassifier`.
-->