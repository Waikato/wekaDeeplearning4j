/*
 * WekaDeeplearning4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * WekaDeeplearning4j is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with WekaDeeplearning4j.  If not, see <https://www.gnu.org/licenses/>.
 *
 * StabilityTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.classifiers.functions;

import lombok.extern.log4j.Log4j2;
import org.junit.Assert;
import org.junit.Test;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.PoolingType;
import weka.dl4j.activations.ActivationIdentity;
import weka.dl4j.activations.ActivationReLU;
import weka.dl4j.activations.ActivationSoftmax;
import weka.dl4j.iterators.instance.AbstractInstanceIterator;
import weka.dl4j.iterators.instance.ConvolutionInstanceIterator;
import weka.dl4j.iterators.instance.DefaultInstanceIterator;
import weka.dl4j.layers.BatchNormalization;
import weka.dl4j.layers.ConvolutionLayer;
import weka.dl4j.layers.DenseLayer;
import weka.dl4j.layers.OutputLayer;
import weka.dl4j.layers.SubsamplingLayer;
import weka.dl4j.updater.Adam;
import weka.util.DatasetLoader;
import weka.util.TestUtil;

/**
 * These tests make sure, that previously achieved performances are stable across version updates.
 * That is, if any of those tests fail it is likely, that code in the configuration does not work as
 * expected, but the code will run nonetheless.
 *
 * @author Steven Lang
 */
@Log4j2
public class StabilityTest {

  public static void evaluate(Dl4jMlpClassifier clf, Instances data, double minPerfomance)
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
    boolean success = testPctCorrect > minPerfomance && trainPctCorrect > minPerfomance;
    log.info("Success: " + success);

    log.info(clf.getModel().conf().toYaml());
    Assert.assertTrue("Performance was < " + minPerfomance + ". TestPctCorrect: " + testPctCorrect
        + ", TrainPctCorrect: " + trainPctCorrect, success);
  }

  @Test
  public void testMnistPerformanceMultiDense() throws Exception {
    Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
    AbstractInstanceIterator ii = new DefaultInstanceIterator();
    ii.setTrainBatchSize(16);
    clf.setInstanceIterator(ii);

    clf.setNumEpochs(100);
    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setL2(0.0005);
    Adam opt = new Adam();
    opt.setLearningRate(0.01);
    nnc.setUpdater(opt);
    nnc.setBiasUpdater(opt);

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
    Adam opt = new Adam();
    opt.setLearningRate(0.01);
    nnc.setUpdater(opt);

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
    Adam opt = new Adam();
    opt.setLearningRate(0.01);
    nnc.setUpdater(opt);

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
    Adam opt = new Adam();
    opt.setLearningRate(0.01);
    nnc.setUpdater(opt);

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
    Adam opt = new Adam();
    opt.setLearningRate(0.01);
    nnc.setUpdater(opt);

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
    Adam opt = new Adam();
    opt.setLearningRate(0.01);
    nnc.setUpdater(opt);

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
    Adam opt = new Adam();
    opt.setLearningRate(0.005);
    nnc.setUpdater(opt);

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
    Adam opt = new Adam();
    opt.setLearningRate(0.01);
    nnc.setUpdater(opt);
    nnc.setBiasUpdater(opt);

    ConvolutionLayer conv = new ConvolutionLayer();
    conv.setNOut(8);

    DenseLayer dl = new DenseLayer();
    dl.setNOut(10);

    SubsamplingLayer sub = new SubsamplingLayer();

    OutputLayer outputLayer = new OutputLayer();

    clf.setSeed(TestUtil.SEED);
    clf.setLayers(conv, sub, dl, outputLayer);
    clf.setNeuralNetConfiguration(nnc);
    clf.setDebug(true);
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
    Adam opt = new Adam();
    opt.setLearningRate(0.01);
    nnc.setUpdater(opt);

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
    Adam opt = new Adam();
    opt.setLearningRate(0.005);
    nnc.setUpdater(opt);

    DenseLayer dl = new DenseLayer();
    dl.setNOut(32);
    BatchNormalization bn = new BatchNormalization();

    OutputLayer outputLayer = new OutputLayer();

    clf.setSeed(TestUtil.SEED);
    clf.setLayers(dl, bn, outputLayer);
    clf.setNeuralNetConfiguration(nnc);

    evaluate(clf, DatasetLoader.loadMiniMnistArff(), 80);
  }

  @Test
  public void testMnistDl4jExample() throws Exception {
    Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
    ConvolutionInstanceIterator cii = new ConvolutionInstanceIterator();
    cii.setTrainBatchSize(16);
    clf.setNumEpochs(20);
    clf.setInstanceIterator(cii);

    weka.dl4j.layers.ConvolutionLayer cl = new weka.dl4j.layers.ConvolutionLayer();
    cl.setNOut(20);
    cl.setStride(new int[]{1, 1});
    cl.setKernelSize(new int[]{5, 5});
    cl.setActivationFunction(new ActivationIdentity());

    weka.dl4j.layers.SubsamplingLayer ssl = new weka.dl4j.layers.SubsamplingLayer();
    ssl.setKernelSize(new int[]{2, 2});
    ssl.setStride(new int[]{2, 2});
    ssl.setPoolingType(PoolingType.MAX);

    weka.dl4j.layers.ConvolutionLayer cl2 = new weka.dl4j.layers.ConvolutionLayer();
    cl2.setNOut(50);
    cl2.setStride(new int[]{1, 1});
    cl2.setKernelSize(new int[]{5, 5});
    cl2.setActivationFunction(new ActivationIdentity());

    weka.dl4j.layers.SubsamplingLayer ssl2 = new weka.dl4j.layers.SubsamplingLayer();
    ssl2.setKernelSize(new int[]{2, 2});
    ssl2.setStride(new int[]{2, 2});
    ssl2.setPoolingType(PoolingType.MAX);

    weka.dl4j.layers.DenseLayer dl = new weka.dl4j.layers.DenseLayer();
    dl.setNOut(500);
    dl.setActivationFunction(new ActivationReLU());

    weka.dl4j.layers.OutputLayer ol = new weka.dl4j.layers.OutputLayer();
    ol.setActivationFunction(new ActivationSoftmax());
    ol.setNOut(10);

    NeuralNetConfiguration conf = new NeuralNetConfiguration();
    final Adam updater = new Adam();
    updater.setLearningRate(0.001);
    conf.setUpdater(updater);
    clf.setNeuralNetConfiguration(conf);
    clf.setLayers(cl, ssl, cl2, ssl2, dl, ol);

    evaluate(clf, DatasetLoader.loadMiniMnistArff(), 80);
  }
}
