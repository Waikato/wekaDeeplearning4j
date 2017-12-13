package weka;

import java.io.File;
import java.io.FileReader;
import org.deeplearning4j.nn.conf.GradientNormalization;
import weka.classifiers.functions.RnnSequenceClassifier;
import weka.core.Instances;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.activations.ActivationTanH;
import weka.dl4j.iterators.instance.sequence.text.TextEmbeddingInstanceIterator;
import weka.dl4j.layers.LSTM;
import weka.dl4j.layers.RnnOutputLayer;

public class Main {
  public static void main(String[] args) throws Exception {
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
    nnc.setUseRegularization(true);
    nnc.setGradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue);
    nnc.setGradientNormalizationThreshold(gradientThreshold);
    nnc.setLearningRate(learningRate);

    // Config classifier
    clf.setLayers(lstm, rnnOut);
    clf.setNeuralNetConfiguration(nnc);
    Instances data = new Instances(new FileReader("src/test/resources/nominal/imdb.arff"));
    data.setClassIndex(1);
    clf.buildClassifier(data);
  }
}
