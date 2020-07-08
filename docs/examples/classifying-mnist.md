# The MNIST Dataset
  
![Mnist Example 0](../img/mnist/img_11854_0.jpg)
![Mnist Example 1](../img/mnist/img_11253_1.jpg)
![Mnist Example 2](../img/mnist/img_10320_2.jpg)
![Mnist Example 3](../img/mnist/img_10324_3.jpg)
![Mnist Example 4](../img/mnist/img_40694_4.jpg)
![Mnist Example 5](../img/mnist/img_10596_5.jpg)
![Mnist Example 6](../img/mnist/img_19625_6.jpg)
![Mnist Example 7](../img/mnist/img_12452_7.jpg)
![Mnist Example 8](../img/mnist/img_10828_8.jpg)
![Mnist Example 9](../img/mnist/img_10239_9.jpg)



The MNIST dataset provides images of handwritten digits of 10 classes (0-9) and suits the task of simple image classification. 

The minimal MNIST arff file can be found in the `datasets/nominal` directory of the WekaDeeplearning4j package. This arff file lists all images in `datasets/nominal/mnist-minimal` and annotates their path with their class label.

**Important note:** The arff dataset contains two features, the first one being the `filename` and the second one being the `class`. Therefore it is necessary to define an `ImageDataSetIterator` in `Dl4jMlpFilter` or `Dl4jMlpClassifier` which uses these filenames in the directory given by the option `-imagesLocation`.

## GUI: LeNet MNIST Evaluation


The first step is to open the MNIST meta ARFF file in the Weka Explorer `Preprocess` tab via `Open File`. A randomly sampled MNIST dataset of 420 images is provided in the WekaDeeplearning4j package for testing purposes (`$WEKA_HOME/packages/wekaDeeplearning4j/datasets/nominal/mnist.meta.minimal.arff`). In the next step, the `Dl4jMlpClassifier` has to be selected as `Classifier` in the `Classify` tab. A click on the classifier will open the configuration window

![Classifier](../img/gui/mlp-classifier.png)

To correctly load the images it is further necessary to select the `Image-Instance-Iterator` as `instance iterator` and point it to the MNIST directory that contains the actual image files (`$WEKA_HOME/packages/wekaDeeplearning4j/datasets/nominal/mnist-minimal/`). 

![Image Instance Iterator](../img/gui/image-instance-iterator.png)

Select `LeNet` from the `zooModel` option as network architecture. 

![LeNet](../img/gui/layer-array.png)

A holdout evaluation strategy has to be selected in the `Test options` box via `Percentage split`, which can be set to 66% for a 2/3 - 1/3 split. The classifier training is now ready to be started with the `Start` button. The resulting classifier evaluation can be examined in the `Classifier output` box. Here, an evaluation summary is shown for the training and testing split. 

The above setup, trained for 50 epochs with a batch size of 256 produces a classification accuracy of 93.71% on the test data after training on the smaller sampled MNIST dataset and a score of 98.57% after training on the full MNIST dataset.

| MNIST   |  Train Size |  Test Size |  Train Accuracy |  Test Accuracy | Train Time      |
| -----   | ----------: | ---------: | --------------: | -------------: | --------------: |
| Sampled |         277 |        143 |          100.0% |         93.71% | 48.99s          |
| Full    |      60.000 |     10.000 |          98.76% |         98.57% | 406.30s         |

Table 1: Results for training time and classification accuracy after 50 epochs for both the sampled and the full MNIST training dataset using the LeNet architecture. Experiments were run on a NVIDIA TITAN X Pascal GPU.


## Commandline
The following run creates a Conv(3x3x8) > Pool(2x2,MAX) > Conv(3x3x8) > Pool(2x2,MAX) > Out architecture
```bash
$ java -Xmx5g weka.Run \
     .Dl4jMlpClassifier \
     -S 1 \
     -iterator ".ImageInstanceIterator -imagesLocation datasets/nominal/mnist-minimal -numChannels 1 -height 28 -width 28 -bs 16" \
     -normalization "Standardize training data" \
     -layer "weka.dl4j.layers.ConvolutionLayer -nFilters 8 -activation weka.dl4j.activations.ActivationReLU -columns 3 -rows 3 -paddingX 0 -paddingY 0 -strideX 1 -strideY 1" \
     -layer "weka.dl4j.layers.SubsamplingLayer -columns 2 -rows 2 -paddingX 0 -paddingY 0 -poolingType MAX -strideX 1 -strideY 1" \
     -layer "weka.dl4j.layers.ConvolutionLayer -nFilters 8 -activation weka.dl4j.activations.ActivationReLU -columns 3 -rows 3 -paddingX 0 -paddingY 0 -strideX 1 -strideY 1" \
     -layer "weka.dl4j.layers.SubsamplingLayer -columns 2 -rows 2 -paddingX 0 -paddingY 0 -poolingType MAX -strideX 1 -strideY 1" \
     -layer "weka.dl4j.layers.OutputLayer -activation weka.dl4j.activations.ActivationSoftmax -lossFn weka.dl4j.lossfunctions.LossMCXENT" \
     -config "weka.dl4j.NeuralNetConfiguration -updater weka.dl4j.updater.Adam" \
     -numEpochs 10 \
     -t datasets/nominal/mnist.meta.minimal.arff \
     -split-percentage 80
```

## Java
The same architecture can be built programmatically with the following Java code

```java
// Load all packages so that Dl4jMlpFilter class can be found using forName("weka.filters.unsupervised.attribute.Dl4jMlpFilter")
weka.core.WekaPackageManager.loadPackages(true);

// Load the dataset
weka.core.Instances data = new weka.core.Instances(new FileReader("datasets/nominal/mnist.meta.minimal.arff"));
data.setClassIndex(data.numAttributes() - 1);
String[] classifierOptions = weka.core.Utils.splitOptions("-iterator \".ImageInstanceIterator -imagesLocation datasets/nominal/mnist-minimal -numChannels 1 -height 28 -width 28 -bs 16\" " +
        "-normalization \"Standardize training data\" -layer \"weka.dl4j.layers.ConvolutionLayer -nFilters 8 -activation weka.dl4j.activations.ActivationReLU -columns 3 -rows 3 -paddingX 0 -paddingY 0 -strideX 1 -strideY 1\" " +
        "-layer \"weka.dl4j.layers.SubsamplingLayer -columns 2 -rows 2 -paddingX 0 -paddingY 0 -poolingType MAX -strideX 1 -strideY 1\" -layer \"weka.dl4j.layers.ConvolutionLayer -nFilters 8 -activation weka.dl4j.activations.ActivationReLU -columns 3 -rows 3 -paddingX 0 -paddingY 0 -strideX 1 -strideY 1\" " +
        "-layer \"weka.dl4j.layers.SubsamplingLayer -columns 2 -rows 2 -paddingX 0 -paddingY 0 -poolingType MAX -strideX 1 -strideY 1\" -layer \"weka.dl4j.layers.OutputLayer -activation weka.dl4j.activations.ActivationSoftmax -lossFn weka.dl4j.lossfunctions.LossMCXENT\" ");
weka.classifiers.AbstractClassifier myClassifier = (AbstractClassifier) weka.core.Utils.forName(weka.classifiers.AbstractClassifier.class, "weka.classifiers.functions.Dl4jMlpClassifier", classifierOptions);

// Stratify and split the data
Random rand = new Random(0);
Instances randData = new Instances(data);
randData.randomize(rand);
randData.stratify(5);
Instances train = randData.trainCV(5, 0);
Instances test = randData.testCV(5, 0);

// Build the classifier on the training data
myClassifier.buildClassifier(train);

// Evaluate the model on test data
Evaluation eval = new Evaluation(test);
eval.evaluateModel(myClassifier, test);

// Output some summary statistics
System.out.println(eval.toSummaryString());
System.out.println(eval.toMatrixString());
```
