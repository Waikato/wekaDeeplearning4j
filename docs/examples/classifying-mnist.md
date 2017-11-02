# The MNIST Dataset
The MNIST dataset provides images of handwritten digits of 10 classes (0-9) and suits the task of simple image classification. 

The minimal MNIST arff file can be found in the `datasets/nominal` directory of the WekaDeeplearning4j package. This arff file lists all images in `datasets/nominal/mnist-minimal` and annotates their path with their class label.

**Important note:** The arff dataset contains two features, the first one being the `filename` and the second one being the `class`. Therefore it is necessary to define an `ImageDataSetIterator` which uses these filenames in the directory given by the option `-imagesLocation`.

## Commandline
The following run creates a Conv(3x3x8)>Pool(2x2,MAX)>Conv(3x3x8)>Pool(2x2,MAX)>Out architecture
```bash
$ java -Xmx5g -cp weka.jar weka.Run \
     .Dl4jMlpClassifier \
     -S 1 \
     -iterator "weka.dl4j.iterators.instance.ImageInstanceIterator -imagesLocation datasets/nominal/mnist-minimal -numChannels 1 -height 28 -width 28 -bs 16" \
     -normalization "Standardize training data" \
     -layer "weka.dl4j.layers.ConvolutionLayer -nFilters 8 -activation weka.dl4j.activations.ActivationReLU -kernelSizeX 3 -kernelSizeY 3 -paddingX 0 -paddingY 0 -strideX 1 -strideY 1" \
     -layer "weka.dl4j.layers.SubsamplingLayer -kernelSizeX 2 -kernelSizeY 2 -paddingX 0 -paddingY 0 -poolingType MAX -strideX 1 -strideY 1" \
     -layer "weka.dl4j.layers.ConvolutionLayer -nFilters 8 -activation weka.dl4j.activations.ActivationReLU -kernelSizeX 3 -kernelSizeY 3 -paddingX 0 -paddingY 0 -strideX 1 -strideY 1" \
     -layer "weka.dl4j.layers.SubsamplingLayer -kernelSizeX 2 -kernelSizeY 2 -paddingX 0 -paddingY 0 -poolingType MAX -strideX 1 -strideY 1" \
     -layer "weka.dl4j.layers.OutputLayer -activation weka.dl4j.activations.ActivationSoftmax -lossFn weka.dl4j.lossfunctions.LossMCXENT -updater ADAM" \
     -numEpochs 10 \
     -t datasets/nominal/mnist.meta.minimal.arff \
     -split-percentage 80
```

## Java
The same architecture can be built programmatically with the following Java code

```java
// Set up the MLP classifier
Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
clf.setSeed(1);
clf.setNumEpochs(10);


// Load the arff file
Instances data = new Instances(new FileReader("datasets/nominal/mnist.meta.minimal.arff"));
data.setClassIndex(data.numAttributes() - 1);


// Load the image iterator
ImageDataSetIterator imgIter = new ImageDataSetIterator();
imgIter.setImagesLocation(new File("datasets/nominal/mnist-minimal"));
imgIter.setHeight(28);
imgIter.setWidth(28);
imgIter.setNumChannels(1);
imgIter.setTrainBatchSize(16);
clf.setDataSetIterator(imgIter);


// Setup the network layers
// First convolution layer, 8 3x3 filter 
ConvolutionLayer convLayer1 = new ConvolutionLayer();
convLayer1.setKernelSizeX(3);
convLayer1.setKernelSizeY(3);
convLayer1.setStrideX(1);
convLayer1.setStrideY(1);
convLayer1.setActivationFn(new ActivationReLU());
convLayer1.setNOut(8);

// First maxpooling layer, 2x2 filter
SubsamplingLayer poolLayer1 = new SubsamplingLayer();
poolLayer1.setPoolingType(PoolingType.MAX);
poolLayer1.setKernelSizeX(2);
poolLayer1.setKernelSizeY(2);
poolLayer1.setStrideX(1);
poolLayer1.setStrideY(1);

// Second convolution layer, 8 3x3 filter
ConvolutionLayer convLayer2 = new ConvolutionLayer();
convLayer2.setKernelSizeX(3);
convLayer2.setKernelSizeY(3);
convLayer2.setStrideX(1);
convLayer2.setStrideY(1);
convLayer2.setActivationFn(new ActivationReLU());
convLayer2.setNOut(8);

// Second maxpooling layer, 2x2 filter
SubsamplingLayer poolLayer2 = new SubsamplingLayer();
poolLayer2.setPoolingType(PoolingType.MAX);
poolLayer2.setKernelSizeX(2);
poolLayer2.setKernelSizeY(2);
poolLayer2.setStrideX(1);
poolLayer2.setStrideY(1);

// Output layer with softmax activation
OutputLayer outputLayer = new OutputLayer();
outputLayer.setActivationFn(new ActivationSoftmax());
outputLayer.setLossFn(new LossMCXENT());


// Set up the network configuration
NeuralNetConfiguration nnc = new NeuralNetConfiguration();
nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
clf.setNeuralNetConfiguration(nnc);


// Set the layers
clf.setLayers(new Layer[]{convLayer1, poolLayer1, convLayer2, poolLayer2, outputLayer});


// Evaluate the network
Evaluation eval = new Evaluation(data);
int numFolds = 10;
eval.crossValidateModel(clf, data, numFolds, new Random(1));

System.out.println("% Correct = " + eval.pctCorrect());
```

