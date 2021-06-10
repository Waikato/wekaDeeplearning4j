# Training an RNN - The IMDB Dataset

As of [ai.stanford.edu](http://ai.stanford.edu/~amaas/data/sentiment/):

> This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well. Raw text and already processed bag of words formats are provided. See the README file contained in the release for more details.

The full IMDB dataset in the ARFF format can be found [here](https://sourceforge.net/projects/weka/files/datasets/text-datasets/imdb-sentiment-2011.arff.gz/download).

## Java RNN
The following code builds a network consisting of an LSTM layer and an RnnOutputLayer, loading imdb reviews and mapping them into a sequence of vectors in the embedding space that is defined by the Google News model. Furthermore, gradient clipping at a value of 1.0 is applied to prevent the network from exploding gradients.

```java
// Download e.g the SLIM Google News model from
// https://github.com/eyaler/word2vec-slim/raw/master/GoogleNews-vectors-negative300-SLIM.bin.gz
final File modelSlim = new File("path/to/google/news/model");

// Setup hyperparameters
final int truncateLength = 80;
final int batchSize = 64;
final int seed = 1;
final int numEpochs = 10;
final int tbpttLength = 20;
final double l2 = 1e-5;
final double gradientThreshold = 1.0;
final double learningRate = 0.02;

// Setup the iterator
TextEmbeddingInstanceIterator tii = new TextEmbeddingInstanceIterator();
tii.setWordVectorLocation(modelSlim);
tii.setTruncateLength(truncateLength);
tii.setTrainBatchSize(batchSize);

// Initialize the classifier
RnnSequenceClassifier clf = new RnnSequenceClassifier();
clf.setSeed(seed);
clf.setNumEpochs(numEpochs);
clf.setInstanceIterator(tii);
clf.settBPTTbackwardLength(tbpttLength);
clf.settBPTTforwardLength(tbpttLength);

// Define the layers
LSTM lstm = new LSTM();
lstm.setNOut(64);
lstm.setActivationFunction(new ActivationTanH());

RnnOutputLayer rnnOut = new RnnOutputLayer();

// Network config
NeuralNetConfiguration nnc = new NeuralNetConfiguration();
nnc.setL2(l2);
nnc.setGradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue);
nnc.setGradientNormalizationThreshold(gradientThreshold);
nnc.setLearningRate(learningRate);

// Config classifier
clf.setLayers(lstm, rnnOut);
clf.setNeuralNetConfiguration(nnc);
Instances data = new Instances(new FileReader("src/test/resources/nominal/imdb.arff"));
data.setClassIndex(1);
clf.buildClassifier(data);
```

## Java CNN
Below is an example building a CNN with two `ConvolutionLayer` that are automatically merged as described [here](../user-guide/nlp.md#using-convolutional-neural-networks) afterwards.
```java

// Embedding vector size
int vectorSize = 300;
int batchSize = 64;

// Create a new Multi-Layer-Perceptron classifier
Dl4jMlpClassifier clf = new Dl4jMlpClassifier();

// Initialize iterator
CnnTextEmbeddingInstanceIterator cnnTextIter = new CnnTextEmbeddingInstanceIterator();
cnnTextIter.setTrainBatchSize(batchSize);
cnnTextIter.setWordVectorLocation(DatasetLoader.loadGoogleNewsVectors());
clf.setInstanceIterator(cnnTextIter);


// Define the layers
// All N convolutional layers will be merged into a single
// output of depth N (simulates multiple inputs)
ConvolutionLayer conv1 = new ConvolutionLayer();
conv1.setKernelSize(new int[] {4, vectorSize});
conv1.setNOut(10);
conv1.setStride(new int[] {1, vectorSize});
conv1.setConvolutionMode(ConvolutionMode.Same);
conv1.setActivationFunction(new ActivationReLU());

ConvolutionLayer conv2 = new ConvolutionLayer();
conv2.setKernelSize(new int[] {3, vectorSize});
conv2.setNOut(10);
conv2.setStride(new int[] {1, vectorSize});
conv2.setConvolutionMode(ConvolutionMode.Same);
conv2.setActivationFunction(new ActivationReLU());

GlobalPoolingLayer gpl = new GlobalPoolingLayer();

OutputLayer out = new OutputLayer();


// Config classifier
clf.setLayers(conv1, conv2, gpl, out);

// Get data
Instances data = new Instances(new FileReader("src/test/resources/nominal/imdb.arff"));
data.setClassIndex(1);

// Build model
clf.buildClassifier(data);
```
