# The Iris Dataset

A very common dataset to test algorithms with is the _Iris Dataset_ . The following explains how to build a neural network from the command line, programmatically in java and in the Weka workbench GUI.

The iris dataset can be found in the `datasets/nominal` directory of the WekaDeeplearning4j package.


<details> 
  <summary>Iris Visualization </summary>
  ![Iris Visualization](../img/iris.png)
</details>

## Commandline
Starting simple, the most straight forward way to create a neural network with this package is by using the commandline. A Single-Layer-Perceptron (the most basic neural network possible) is shown in the following
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
outputLayer.setActivationFunction(new ActivationSoftmax());
outputLayer.setLossFn(new LossMCXENT());

NeuralNetConfiguration nnc = new NeuralNetConfiguration();
nnc.setUpdater(new Adam());

// Add the layers to the classifier
clf.setLayers(new Layer[]{outputLayer});
clf.setNeuralNetConfiguration(nnc);

// Evaluate the network
Evaluation eval = new Evaluation(data);
int numFolds = 10;
eval.crossValidateModel(clf, data, numFolds, new Random(1));

System.out.println(eval.toSummaryString());
```

