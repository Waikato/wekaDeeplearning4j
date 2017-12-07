package weka.classifiers.functions;

import lombok.extern.slf4j.Slf4j;
import org.junit.Assert;
import org.junit.Test;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.iterators.instance.AbstractInstanceIterator;
import weka.dl4j.iterators.instance.ConvolutionInstanceIterator;
import weka.dl4j.iterators.instance.DefaultInstanceIterator;
import weka.dl4j.layers.BatchNormalization;
import weka.dl4j.layers.ConvolutionLayer;
import weka.dl4j.layers.DenseLayer;
import weka.dl4j.layers.OutputLayer;
import weka.dl4j.layers.SubsamplingLayer;
import weka.util.DatasetLoader;
import weka.util.TestUtil;

/**
 * These tests make sure, that previously achieved performances are stable across version updates.
 * That is, if any of those tests fail it is likely, that code in the configuration does not work as
 * expected, but the code will run nonetheless.
 *
 * @author Steven Lang
 */
@Slf4j
public class StabilityTest {

  public static void evaluate(Classifier clf, Instances data, double minPerfomance)
      throws Exception {
    Instances[] split = TestUtil.splitTrainTest(data);

    Instances train = split[0];
    Instances test = split[1];

    clf.buildClassifier(train);
    Evaluation trainEval = new Evaluation(train);
    trainEval.evaluateModel(clf, train);

    Evaluation testEval = new Evaluation(train);
    testEval.evaluateModel(clf, test);

    final double testPctCorrect = testEval.pctCorrect();
    final double trainPctCorrect = trainEval.pctCorrect();

    log.info("Train: {}, Test: {}", trainPctCorrect, testPctCorrect);
    boolean success =
        testPctCorrect > minPerfomance && trainPctCorrect > minPerfomance;
    Assert.assertTrue(success);
  }

  @Test
  public void testMnistPerformanceMultiDense() throws Exception {
    Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
    AbstractInstanceIterator ii = new DefaultInstanceIterator();
    ii.setTrainBatchSize(16);
    clf.setInstanceIterator(ii);

    clf.setNumEpochs(20);
    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setLearningRate(0.01);

    DenseLayer denseLayer = new DenseLayer();
    denseLayer.setNOut(32);

    DenseLayer denseLayer2 = new DenseLayer();
    denseLayer2.setNOut(16);

    OutputLayer outputLayer = new OutputLayer();

    clf.setSeed(TestUtil.SEED);
    clf.setLayers(denseLayer, denseLayer2, outputLayer);
    clf.setNeuralNetConfiguration(nnc);

    evaluate(clf, DatasetLoader.loadMiniMnistArff(), 80);
  }

  @Test
  public void testMnistPerformanceSingleDense() throws Exception {
    Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
    AbstractInstanceIterator ii = new DefaultInstanceIterator();
    ii.setTrainBatchSize(16);
    clf.setInstanceIterator(ii);

    clf.setNumEpochs(20);
    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setLearningRate(0.01);

    DenseLayer denseLayer = new DenseLayer();
    denseLayer.setNOut(32);

    OutputLayer outputLayer = new OutputLayer();

    clf.setSeed(TestUtil.SEED);
    clf.setLayers(denseLayer, outputLayer);
    clf.setNeuralNetConfiguration(nnc);

    evaluate(clf, DatasetLoader.loadMiniMnistArff(), 80);
  }

  @Test
  public void testMnistPerformanceConv() throws Exception {
    Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
    ConvolutionInstanceIterator cii = new ConvolutionInstanceIterator();
    cii.setTrainBatchSize(16);
    clf.setNumEpochs(20);
    clf.setInstanceIterator(cii);

    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setLearningRate(0.01);

    ConvolutionLayer conv = new ConvolutionLayer();
    conv.setNOut(8);

    OutputLayer outputLayer = new OutputLayer();

    clf.setSeed(TestUtil.SEED);
    clf.setLayers(conv, outputLayer);
    clf.setNeuralNetConfiguration(nnc);

    evaluate(clf, DatasetLoader.loadMiniMnistArff(), 80);
  }

  @Test
  public void testMnistPerformanceConvSub() throws Exception {
    Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
    ConvolutionInstanceIterator cii = new ConvolutionInstanceIterator();
    cii.setTrainBatchSize(16);
    clf.setNumEpochs(20);
    clf.setInstanceIterator(cii);

    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setLearningRate(0.01);

    ConvolutionLayer conv = new ConvolutionLayer();
    conv.setNOut(8);

    SubsamplingLayer sub = new SubsamplingLayer();

    OutputLayer outputLayer = new OutputLayer();

    clf.setSeed(TestUtil.SEED);
    clf.setLayers(conv, sub, outputLayer);
    clf.setNeuralNetConfiguration(nnc);

    evaluate(clf, DatasetLoader.loadMiniMnistArff(), 80);
  }

  @Test
  public void testMnistPerformanceConvConv() throws Exception {
    Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
    ConvolutionInstanceIterator cii = new ConvolutionInstanceIterator();
    cii.setTrainBatchSize(16);
    clf.setNumEpochs(20);
    clf.setInstanceIterator(cii);

    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setLearningRate(0.01);

    ConvolutionLayer conv = new ConvolutionLayer();
    conv.setNOut(8);

    ConvolutionLayer conv2 = new ConvolutionLayer();
    conv2.setNOut(8);

    OutputLayer outputLayer = new OutputLayer();

    clf.setSeed(TestUtil.SEED);
    clf.setLayers(conv, conv2, outputLayer);
    clf.setNeuralNetConfiguration(nnc);

    evaluate(clf, DatasetLoader.loadMiniMnistArff(), 80);
  }

  @Test
  public void testMnistPerformanceConvSubConv() throws Exception {
    Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
    ConvolutionInstanceIterator cii = new ConvolutionInstanceIterator();
    cii.setTrainBatchSize(16);
    clf.setNumEpochs(20);
    clf.setInstanceIterator(cii);

    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setLearningRate(0.01);

    ConvolutionLayer conv = new ConvolutionLayer();
    conv.setNOut(8);
    SubsamplingLayer sub = new SubsamplingLayer();

    ConvolutionLayer conv2 = new ConvolutionLayer();
    conv2.setNOut(8);

    OutputLayer outputLayer = new OutputLayer();

    clf.setSeed(TestUtil.SEED);
    clf.setLayers(conv, sub, conv2, outputLayer);
    clf.setNeuralNetConfiguration(nnc);

    evaluate(clf, DatasetLoader.loadMiniMnistArff(), 80);
  }

  @Test
  public void testMnistPerformanceConvDense() throws Exception {
    Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
    ConvolutionInstanceIterator cii = new ConvolutionInstanceIterator();
    cii.setTrainBatchSize(16);
    clf.setNumEpochs(20);
    clf.setInstanceIterator(cii);

    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setLearningRate(0.005);

    ConvolutionLayer conv = new ConvolutionLayer();
    conv.setNOut(16);

    DenseLayer dl = new DenseLayer();
    dl.setNOut(32);

    OutputLayer outputLayer = new OutputLayer();

    clf.setSeed(TestUtil.SEED);
    clf.setLayers(conv, dl, outputLayer);
    clf.setNeuralNetConfiguration(nnc);

    evaluate(clf, DatasetLoader.loadMiniMnistArff(), 80);
  }

  @Test
  public void testMnistPerformanceConvSubDense() throws Exception {
    Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
    ConvolutionInstanceIterator cii = new ConvolutionInstanceIterator();
    cii.setTrainBatchSize(16);
    clf.setNumEpochs(20);
    clf.setInstanceIterator(cii);

    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setLearningRate(0.01);

    ConvolutionLayer conv = new ConvolutionLayer();
    conv.setNOut(8);

    DenseLayer dl = new DenseLayer();
    dl.setNOut(10);

    SubsamplingLayer sub = new SubsamplingLayer();

    OutputLayer outputLayer = new OutputLayer();

    clf.setSeed(TestUtil.SEED);
    clf.setLayers(conv, sub, dl, outputLayer);
    clf.setNeuralNetConfiguration(nnc);

    evaluate(clf, DatasetLoader.loadMiniMnistArff(), 80);
  }

  @Test
  public void testMnistPerformanceConvBN() throws Exception {
    Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
    ConvolutionInstanceIterator cii = new ConvolutionInstanceIterator();
    cii.setTrainBatchSize(16);
    clf.setNumEpochs(20);
    clf.setInstanceIterator(cii);

    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setLearningRate(0.01);

    ConvolutionLayer conv = new ConvolutionLayer();
    conv.setNOut(8);
    BatchNormalization bn = new BatchNormalization();

    OutputLayer outputLayer = new OutputLayer();

    clf.setSeed(TestUtil.SEED);
    clf.setLayers(conv, bn, outputLayer);
    clf.setNeuralNetConfiguration(nnc);

    evaluate(clf, DatasetLoader.loadMiniMnistArff(), 80);
  }

  @Test
  public void testMnistPerformanceDenseBN() throws Exception {
    Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
    ConvolutionInstanceIterator cii = new ConvolutionInstanceIterator();
    cii.setTrainBatchSize(16);
    clf.setNumEpochs(20);
    clf.setInstanceIterator(cii);

    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setLearningRate(0.005);

    DenseLayer dl = new DenseLayer();
    dl.setNOut(32);
    BatchNormalization bn = new BatchNormalization();

    OutputLayer outputLayer = new OutputLayer();

    clf.setSeed(TestUtil.SEED);
    clf.setLayers(dl, bn, outputLayer);
    clf.setNeuralNetConfiguration(nnc);

    evaluate(clf, DatasetLoader.loadMiniMnistArff(), 80);
  }

}
