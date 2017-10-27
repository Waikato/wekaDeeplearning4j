# The Iris Dataset
A very common dataset to test algorithms with is the _Iris Dataset_ . The following explains how to build a neural network from the command line, programmatically in java and in the Weka workbench GUI.

The iris dataset can be found in the `datasets/nominal` directory of the WekaDeeplearning4j package.
## Commandline
Starting simple, the most straight forward way to create a neural network with this package is by using the commandline. A Single-Layer-Perceptron (the most basic neural network possible) is shown in the following
```bash
$ java -cp weka.jar weka.Run \
       .Dl4jMlpClassifier \
       -S 1 \ 
       -layer "weka.dl4j.layers.OutputLayer \
              -activation weka.dl4j.activations.ActivationSoftmax \
              -updater SGD \
              -lr 0.01 \
              -blr 0.01 \
              -name \"Output layer\" \
              -lossFn weka.dl4j.lossfunctions.LossMCXENT" \
       -numEpochs 10 \
       -t datasets/nominal/iris.arff \
       -split-percentage 66
```


## Java
The same architecture can be built programmatically with the following Java code

```java
// Create a new Multi-Layer-Perceptron classifier
Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
// Set a seed for reproducable results
clf.setSeed(1);

// Load the iris dataset and set its class index
Instances data = new Instances(new FileReader("datasets/nominal/iris.arff"));
data.setClassIndex(data.numAttributes() - 1);

// Define the output layer
OutputLayer outputLayer = new OutputLayer();
outputLayer.setActivationFn(new ActivationSoftmax());
outputLayer.setUpdater(Updater.SGD);
outputLayer.setLearningRate(0.01);
outputLayer.setBiasLearningRate(0.01);
outputLayer.setLossFn(new LossMCXENT());

// Add the layers to the classifier
clf.setLayers(new Layer[]{outputLayer});

// Evaluate the network
Evaluation eval = new Evaluation(data);
int numFolds = 10;
eval.crossValidateModel(clf, data, numFolds, new Random(1));

System.out.println(eval.toSummaryString());
```

