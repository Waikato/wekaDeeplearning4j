# The Iris Dataset

A very common dataset to test algorithms with is the _Iris Dataset_ . The following explains how to build a neural network from the command line, programmatically in java and in the Weka workbench GUI.

The iris dataset can be found in the `datasets/nominal` directory of the WekaDeeplearning4j package.

Iris Visualization ![Iris Visualization](../img/iris.png)

## Commandline
Starting simple, the most straight forward way to create a neural network with this package is by using the commandline. A Single-Layer-Perceptron (the most basic neural network possible) is shown in the following
```bash
$ java weka.Run \
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
// Load all packages so that Dl4jMlpFilter class can be found using forName("weka.filters.unsupervised.attribute.Dl4jMlpFilter")
weka.core.WekaPackageManager.loadPackages(true);

// Load the dataset
weka.core.Instances data = new weka.core.Instances(new FileReader("datasets/nominal/iris.arff"));
data.setClassIndex(data.numAttributes() - 1);
String[] classifierOptions = weka.core.Utils.splitOptions("-S 1 -numEpochs 10 -layer \"weka.dl4j.layers.OutputLayer -activation weka.dl4j.activations.ActivationSoftmax -lossFn weka.dl4j.lossfunctions.LossMCXENT\"");
weka.classifiers.AbstractClassifier myClassifier = (AbstractClassifier) weka.core.Utils.forName(weka.classifiers.AbstractClassifier.class, "weka.classifiers.functions.Dl4jMlpClassifier", classifierOptions);

// Stratify and split the data
Random rand = new Random(0);
Instances randData = new Instances(data);
randData.randomize(rand);
randData.stratify(3);
Instances train = randData.trainCV(3, 0);
Instances test = randData.testCV(3, 0);

// Build the classifier on the training data
myClassifier.buildClassifier(train);

// Evaluate the model on test data
Evaluation eval = new Evaluation(test);
eval.evaluateModel(myClassifier, test);

// Output some summary statistics
System.out.println(eval.toSummaryString());
System.out.println(eval.toMatrixString());
```

