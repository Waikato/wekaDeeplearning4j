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
 * LayerTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j.layers;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Assert;
import org.junit.Test;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.Instances;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.PoolingType;
import weka.dl4j.activations.ActivationReLU;
import weka.dl4j.activations.ActivationSoftmax;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.util.DatasetLoader;

/**
 * JUnit tests for different layers {@link weka.dl4j.layers}
 *
 * @author Steven Lang
 */
public class LayerTest {

  /**
   * Test subsampling layer.
   */
  @Test
  public void testSubsamplingLayer() throws Exception {
    // CLF
    Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
    clf.setSeed(1);

    // Data
    Instances data = DatasetLoader.loadMiniMnistMeta();
    data.setClassIndex(data.numAttributes() - 1);
    final ImageInstanceIterator imgIter = DatasetLoader.loadMiniMnistImageIterator();
    clf.setInstanceIterator(imgIter);

    SubsamplingLayer pool = new SubsamplingLayer();
    pool.setKernelSizeX(2);
    pool.setKernelSizeY(2);
    pool.setPoolingType(PoolingType.MAX);

    OutputLayer outputLayer = new OutputLayer();
    outputLayer.setActivationFunction(new ActivationSoftmax());

    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
    nnc.setWeightInit(WeightInit.XAVIER);

    clf.setNeuralNetConfiguration(nnc);
    clf.setLayers(pool, outputLayer);

    clf.setNumEpochs(1);
    clf.buildClassifier(data);
    final double[][] res = clf.distributionsForInstances(data);
    Assert.assertEquals(DatasetLoader.NUM_INSTANCES_MNIST, res.length);
    Assert.assertEquals(DatasetLoader.NUM_CLASSES_MNIST, res[0].length);
  }

  /**
   * Test a no-hidden-layer neural net (i.e. a perceptron) on the numeric diabetes dataset
   *
   * @throws Exception the exception
   */
  @Test
  public void testOutputLayer() throws Exception {
    Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
    OutputLayer out = new OutputLayer();
    clf.setLayers(out);
    Instances data = DatasetLoader.loadDiabetes();

    clf.setNumEpochs(1);
    clf.buildClassifier(data);
    final double[][] res = clf.distributionsForInstances(data);
    Assert.assertEquals(DatasetLoader.NUM_INSTANCES_DIABETES, res.length);
    Assert.assertEquals(DatasetLoader.NUM_CLASSES_DIABETES, res[0].length);
  }

  /**
   * Test batchnorm layer.
   */
  @Test
  public void testBatchNormLayer() throws Exception {
    // CLF
    Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
    clf.setSeed(1);

    // Data
    Instances data = DatasetLoader.loadMiniMnistMeta();
    data.setClassIndex(data.numAttributes() - 1);
    final ImageInstanceIterator imgIter = DatasetLoader.loadMiniMnistImageIterator();
    clf.setInstanceIterator(imgIter);

    DenseLayer dl = new DenseLayer();
    dl.setNOut(8);

    BatchNormalization bn = new BatchNormalization();
    OutputLayer outputLayer = new OutputLayer();
    outputLayer.setActivationFunction(new ActivationSoftmax());

    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
    nnc.setWeightInit(WeightInit.XAVIER);
    clf.setNeuralNetConfiguration(nnc);
    clf.setLayers(dl, bn, outputLayer);

    clf.setNumEpochs(1);
    clf.buildClassifier(data);
    double[][] res = clf.distributionsForInstances(data);
    Assert.assertEquals(DatasetLoader.NUM_INSTANCES_MNIST, res.length);
    Assert.assertEquals(DatasetLoader.NUM_CLASSES_MNIST, res[0].length);
  }

  /**
   * Test batchnorm layer.
   */
  @Test
  public void testConvolutionalLayer() throws Exception {
    // CLF
    Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
    clf.setSeed(1);
    clf.setNumEpochs(1);

    // Data
    Instances data = DatasetLoader.loadMiniMnistMeta();
    final ImageInstanceIterator imgIter = DatasetLoader.loadMiniMnistImageIterator();
    clf.setInstanceIterator(imgIter);

    ConvolutionLayer convLayer = new ConvolutionLayer();
    convLayer.setKernelSize(new int[]{3, 3});
    convLayer.setActivationFunction(new ActivationReLU());
    convLayer.setNOut(32);
    OutputLayer outputLayer = new OutputLayer();
    outputLayer.setActivationFunction(new ActivationSoftmax());

    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
    nnc.setWeightInit(WeightInit.XAVIER);

    clf.setNumEpochs(1);
    clf.setNeuralNetConfiguration(nnc);
    clf.setLayers(convLayer, outputLayer);

    clf.buildClassifier(data);
    double[][] res = clf.distributionsForInstances(data);
    Assert.assertEquals(DatasetLoader.NUM_INSTANCES_MNIST, res.length);
    Assert.assertEquals(DatasetLoader.NUM_CLASSES_MNIST, res[0].length);
  }
}
