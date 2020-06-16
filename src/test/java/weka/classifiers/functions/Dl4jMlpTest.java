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
 * Dl4jMlpTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.classifiers.functions;

import static org.junit.Assert.assertEquals;
import static weka.util.TestUtil.readClf;
import static weka.util.TestUtil.saveClf;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestName;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.core.Instances;
import weka.core.InvalidNetworkArchitectureException;
import weka.core.MissingOutputLayerException;
import weka.core.WrongIteratorException;
import weka.dl4j.CacheMode;
import weka.dl4j.ConvolutionMode;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.PoolingType;
import weka.dl4j.activations.ActivationIdentity;
import weka.dl4j.activations.ActivationReLU;
import weka.dl4j.activations.ActivationSoftmax;
import weka.dl4j.dropout.Dropout;
import weka.dl4j.earlystopping.EarlyStopping;
import weka.dl4j.iterators.instance.ConvolutionInstanceIterator;
import weka.dl4j.iterators.instance.DefaultInstanceIterator;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.iterators.instance.sequence.text.cnn.CnnTextEmbeddingInstanceIterator;
import weka.dl4j.iterators.instance.sequence.text.cnn.CnnTextFilesEmbeddingInstanceIterator;
import weka.dl4j.layers.BatchNormalization;
import weka.dl4j.layers.ConvolutionLayer;
import weka.dl4j.layers.DenseLayer;
import weka.dl4j.layers.GlobalPoolingLayer;
import weka.dl4j.layers.Layer;
import weka.dl4j.layers.OutputLayer;
import weka.dl4j.layers.SubsamplingLayer;
import weka.dl4j.listener.EpochListener;
import weka.dl4j.lossfunctions.LossMCXENT;
import weka.dl4j.lossfunctions.LossMSE;
import weka.dl4j.updater.Adam;
import weka.dl4j.zoo.Dl4jLeNet;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.util.DatasetLoader;
import weka.util.TestUtil;

/**
 * JUnit tests for the Dl4jMlpClassifier. Tests nominal classes with iris, numerical classes with
 * diabetes and image classification with minimal mnist.
 *
 * @author Steven Lang
 */
public class Dl4jMlpTest {

  /**
   * Logger instance
   */
  private static final Logger logger = LoggerFactory.getLogger(Dl4jMlpTest.class);
  /**
   * Current name
   */
  @Rule
  public TestName name = new TestName();
  /**
   * Classifier
   */
  private Dl4jMlpClassifier clf;
  /**
   * Dataset mnist
   */
  private Instances dataMnist;
  /**
   * Mnist image loader
   */
  private ImageInstanceIterator idiMnist;
  /**
   * Dataset iris
   */
  private Instances dataIris;
  /**
   * Start time for time measurement
   */
  private long startTime;

  @Before
  public void before() throws Exception {
    // Init mlp clf
    clf = new Dl4jMlpClassifier();
    clf.setSeed(TestUtil.SEED);
    clf.setNumEpochs(TestUtil.DEFAULT_NUM_EPOCHS);

    clf.setDebug(false);

    // Init data
    dataMnist = DatasetLoader.loadMiniMnistMeta();
    idiMnist = DatasetLoader.loadMiniMnistImageIterator();
    idiMnist.setTrainBatchSize(TestUtil.DEFAULT_BATCHSIZE);
    dataIris = DatasetLoader.loadIris();
    startTime = System.currentTimeMillis();
    //        TestUtil.enableUIServer(clf);
  }

  @After
  public void after() {

    //        logger.info("Press anything to close");
    //        Scanner sc = new Scanner(System.in);
    //        sc.next();
    double time = (System.currentTimeMillis() - startTime) / 1000.0;
    logger.info("Testmethod: " + name.getMethodName());
    logger.info("Time: " + time + "s");
  }

  /**
   * Test minimal mnist conv net.
   *
   * @throws Exception IO error.
   */
  @Test
  public void testMinimalMnistConvNet() throws Exception {
    clf.setInstanceIterator(idiMnist);

    int[] threeByThree = {3, 3};
    int[] twoByTwo = {2, 2};
    int[] oneByOne = {1, 1};
    List<Layer> layers = new ArrayList<>();

    ConvolutionLayer convLayer1 = new ConvolutionLayer();
    convLayer1.setKernelSize(threeByThree);
    convLayer1.setStride(oneByOne);
    convLayer1.setNOut(8);
    convLayer1.setLayerName("Conv-layer 1");
    layers.add(convLayer1);

    SubsamplingLayer poolLayer1 = new SubsamplingLayer();
    poolLayer1.setPoolingType(PoolingType.MAX);
    poolLayer1.setKernelSize(twoByTwo);
    poolLayer1.setLayerName("Pool1");
    layers.add(poolLayer1);

    ConvolutionLayer convLayer3 = new ConvolutionLayer();
    convLayer3.setNOut(8);
    convLayer3.setKernelSize(threeByThree);
    layers.add(convLayer3);

    BatchNormalization bn4 = new BatchNormalization();
    bn4.setActivationFunction(new ActivationReLU());
    layers.add(bn4);

    SubsamplingLayer poolLayer2 = new SubsamplingLayer();
    poolLayer2.setPoolingType(PoolingType.MAX);
    poolLayer2.setKernelSize(twoByTwo);
    layers.add(poolLayer2);

    OutputLayer outputLayer = new OutputLayer();
    outputLayer.setActivationFunction(new ActivationSoftmax());
    outputLayer.setLossFn(new LossMCXENT());
    layers.add(outputLayer);

    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);

    clf.setNeuralNetConfiguration(nnc);
    Layer[] ls = new Layer[layers.size()];
    layers.toArray(ls);
    clf.setLayers(ls);

    TestUtil.holdout(clf, dataMnist);
  }

  /**
   * Test minimal mnist dense net.
   *
   * @throws Exception IO error.
   */
  @Test
  public void testMinimalMnistDense() throws Exception {
    clf.setInstanceIterator(idiMnist);

    DenseLayer denseLayer = new DenseLayer();
    denseLayer.setNOut(128);
    denseLayer.setActivationFunction(new ActivationReLU());

    DenseLayer denseLayer2 = new DenseLayer();
    denseLayer2.setNOut(32);
    denseLayer2.setActivationFunction(new ActivationReLU());

    OutputLayer outputLayer = new OutputLayer();
    outputLayer.setActivationFunction(new ActivationSoftmax());
    outputLayer.setLossFn(new LossMCXENT());

    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
    nnc.setSeed(TestUtil.SEED);

    clf.setNeuralNetConfiguration(nnc);
    clf.setLayers(denseLayer, denseLayer2, outputLayer);
    clf.setIterationListener(new EpochListener());
    TestUtil.holdout(clf, dataMnist);
  }

  /**
   * Test Layer setup DENSE to CONV which is currently not supported
   *
   * @throws Exception Could not build classifier.
   */
  @Test(expected = RuntimeException.class)
  public void testIllegalIrisConv() throws Exception {
    final ConvolutionInstanceIterator it = new ConvolutionInstanceIterator();
    it.setHeight(1);
    it.setWidth(4);
    clf.setInstanceIterator(it);

    ConvolutionLayer cl = new ConvolutionLayer();
    cl.setNOut(3);
    cl.setKernelSize(new int[]{1, 1});
    cl.setStride(new int[]{1, 1});

    DenseLayer dl = new DenseLayer();
    dl.setNOut(10);

    OutputLayer ol = new OutputLayer();
    clf.setLayers(dl, cl, ol);
    clf.buildClassifier(dataIris);
  }

  /**
   * Test convolution while setting {@link DefaultInstanceIterator} which is forbidden.
   *
   * @throws Exception Could not build classifier.
   */
  @Test(expected = RuntimeException.class)
  public void testIllegalIrisConvDefaultInstanceIterator() throws Exception {
    // DefaultInstanceIterator should not be allowed in the combination with
    // convolutional layers.
    clf.setInstanceIterator(new DefaultInstanceIterator());

    DenseLayer dl = new DenseLayer();
    dl.setNOut(10);

    ConvolutionLayer cl = new ConvolutionLayer();
    cl.setNOut(3);
    cl.setKernelSize(new int[]{1, 1});
    cl.setStride(new int[]{1, 1});

    OutputLayer ol = new OutputLayer();
    clf.setLayers(cl, dl, ol);
    clf.buildClassifier(dataIris);
  }

  /**
   * Test iris convolution.
   *
   * @throws Exception Could not build classifier.
   */
  @Test
  public void testIrisConv() throws Exception {
    final ConvolutionInstanceIterator it = new ConvolutionInstanceIterator();
    it.setHeight(1);
    it.setWidth(4);
    clf.setInstanceIterator(it);

    DenseLayer dl = new DenseLayer();
    dl.setNOut(10);

    ConvolutionLayer cl = new ConvolutionLayer();
    cl.setNOut(3);
    cl.setKernelSize(new int[]{1, 1});
    cl.setStride(new int[]{1, 1});

    OutputLayer ol = new OutputLayer();
    clf.setLayers(cl, dl, ol);
    clf.buildClassifier(dataIris);
  }

  /**
   * Test to serialization of the classifier. This is important for the GUI usage.
   *
   * @throws Exception Could not build classifier.
   */
  @Test
  public void testSerialization() throws Exception {
    clf.setInstanceIterator(idiMnist);

    DenseLayer denseLayer = new DenseLayer();
    denseLayer.setNOut(8);
    denseLayer.setLayerName("Dense-layer");
    denseLayer.setActivationFunction(new ActivationReLU());

    DenseLayer denseLayer2 = new DenseLayer();
    denseLayer2.setNOut(4);
    denseLayer2.setLayerName("Dense-layer");
    denseLayer2.setActivationFunction(new ActivationReLU());

    OutputLayer outputLayer = new OutputLayer();
    outputLayer.setActivationFunction(new ActivationSoftmax());
    outputLayer.setLossFn(new LossMCXENT());
    outputLayer.setLayerName("Output-layer");

    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
    nnc.setSeed(TestUtil.SEED);

    clf.setNeuralNetConfiguration(nnc);
    clf.setLayers(denseLayer, denseLayer2, outputLayer);
    clf.setIterationListener(new EpochListener());

    File out = Paths.get(System.getProperty("java.io.tmpdir"), "out.object").toFile();
    ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(out));
    clf.initializeClassifier(dataMnist);
    oos.writeObject(clf);

    ObjectInputStream ois = new ObjectInputStream(new FileInputStream(out));
    Dl4jMlpClassifier clf2 = (Dl4jMlpClassifier) ois.readObject();

    clf2.setNumEpochs(1);
    clf2.buildClassifier(dataMnist);
  }

  /**
   * Test no outputlayer
   */
  @Test(expected = MissingOutputLayerException.class)
  public void testLastLayerNoOutputLayer() throws Exception {
    clf.setLayers(new DenseLayer());
    clf.initializeClassifier(dataIris);
  }

  /**
   * Test async iterator
   */
  @Test
  public void testAsyncIterator() throws Exception {
    clf.setQueueSize(4);
    clf.buildClassifier(dataIris);
  }

  /**
   * Test zoo model with wrong iterator
   */
  @Test(expected = WrongIteratorException.class)
  public void testZooModelWithoutImageIterator() throws Exception {
    clf.setZooModel(new Dl4jLeNet());
    clf.setInstanceIterator(new DefaultInstanceIterator());
    clf.buildClassifier(dataIris);
  }

  @Test
  public void testConjugateGradientDescent() throws Exception {
    DenseLayer dl1 = new DenseLayer();
    dl1.setNOut(16);

    OutputLayer ol = new OutputLayer();

    Layer[] ls = new Layer[]{dl1, ol};

    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setOptimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT);
    nnc.setSeed(TestUtil.SEED);
    clf.setNeuralNetConfiguration(nnc);

    clf.setNumEpochs(3);
    clf.setLayers(ls);
    final EarlyStopping config = new EarlyStopping(0, 0);
    clf.setEarlyStopping(config);
    TestUtil.crossValidate(clf, DatasetLoader.loadGlass());
  }

  @Test
  public void testTextCnnClassification() throws Exception {
    CnnTextEmbeddingInstanceIterator cnnTextIter = new CnnTextEmbeddingInstanceIterator();
    cnnTextIter.setTrainBatchSize(128);
    cnnTextIter.setWordVectorLocation(DatasetLoader.loadGoogleNewsVectors());
    clf.setInstanceIterator(cnnTextIter);

    cnnTextIter.initialize();
    final WordVectors wordVectors = cnnTextIter.getWordVectors();
    int vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;

    ConvolutionLayer conv1 = new ConvolutionLayer();
    conv1.setKernelSize(new int[]{4, vectorSize});
    conv1.setNOut(10);
    conv1.setStride(new int[]{1, vectorSize});
    conv1.setConvolutionMode(ConvolutionMode.Same);
    conv1.setActivationFunction(new ActivationReLU());

    BatchNormalization bn1 = new BatchNormalization();

    ConvolutionLayer conv2 = new ConvolutionLayer();
    conv2.setKernelSize(new int[]{3, vectorSize});
    conv2.setNOut(10);
    conv2.setStride(new int[]{1, vectorSize});
    conv2.setConvolutionMode(ConvolutionMode.Same);
    conv2.setActivationFunction(new ActivationReLU());

    BatchNormalization bn2 = new BatchNormalization();

    ConvolutionLayer conv3 = new ConvolutionLayer();
    conv3.setKernelSize(new int[]{2, vectorSize});
    conv3.setNOut(10);
    conv3.setStride(new int[]{1, vectorSize});
    conv3.setConvolutionMode(ConvolutionMode.Same);
    conv3.setActivationFunction(new ActivationReLU());

    BatchNormalization bn3 = new BatchNormalization();

    GlobalPoolingLayer gpl = new GlobalPoolingLayer();

    OutputLayer out = new OutputLayer();

    //    clf.setLayers(conv1, bn1, conv2, bn2, conv3, bn3, gpl, out);
    clf.setLayers(conv1, conv2, conv3, gpl, out);
    //    clf.setNumEpochs(50);
    clf.setCacheMode(CacheMode.MEMORY);
    final EpochListener l = new EpochListener();
    l.setN(1);
    clf.setIterationListener(l);

    clf.setEarlyStopping(new EarlyStopping(10, 15));
    clf.setDebug(true);

    // NNC
    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setL2(1e-3);
    final Dropout dropout = new Dropout();
    dropout.setP(0.2);
    nnc.setDropout(dropout);
    clf.setNeuralNetConfiguration(nnc);

    // Data
    final Instances data = DatasetLoader.loadImdb();
    data.randomize(new Random(42));
    RemovePercentage rp = new RemovePercentage();
    rp.setInputFormat(data);
    rp.setPercentage(98);
    final Instances dataFiltered = Filter.useFilter(data, rp);

    TestUtil.holdout(clf, dataFiltered);
  }

  @Test
  public void testTextCnnRegression() throws Exception {
    CnnTextEmbeddingInstanceIterator cnnTextIter = new CnnTextEmbeddingInstanceIterator();
    cnnTextIter.setTrainBatchSize(64);
    cnnTextIter.setWordVectorLocation(DatasetLoader.loadGoogleNewsVectors());
    clf.setInstanceIterator(cnnTextIter);

    cnnTextIter.initialize();
    final WordVectors wordVectors = cnnTextIter.getWordVectors();
    int vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;

    ConvolutionLayer conv1 = new ConvolutionLayer();
    conv1.setKernelSize(new int[]{3, vectorSize});
    conv1.setNOut(10);
    conv1.setStride(new int[]{1, vectorSize});
    conv1.setConvolutionMode(ConvolutionMode.Same);

    ConvolutionLayer conv2 = new ConvolutionLayer();
    conv2.setKernelSize(new int[]{2, vectorSize});
    conv2.setNOut(10);
    conv2.setStride(new int[]{1, vectorSize});
    conv2.setConvolutionMode(ConvolutionMode.Same);

    GlobalPoolingLayer gpl = new GlobalPoolingLayer();

    OutputLayer out = new OutputLayer();
    out.setLossFn(new LossMSE());
    out.setActivationFunction(new ActivationIdentity());

    clf.setLayers(conv1, conv2, gpl, out);
    //    clf.setNumEpochs(200);
    clf.setCacheMode(CacheMode.MEMORY);
    final EpochListener l = new EpochListener();
    l.setN(20);
    clf.setIterationListener(l);
    clf.setDebug(true);
    clf.setEarlyStopping(new EarlyStopping(0, 10));
    final Instances data = DatasetLoader.loadAnger();

    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setL2(0.00001);
    Adam opt = new Adam();
    opt.setLearningRate(0.001);
    nnc.setUpdater(opt);

    clf.setNeuralNetConfiguration(nnc);
    TestUtil.holdout(clf, data);
  }

  @Test
  public void testTextCnnTextFilesRegression() throws Exception {
    CnnTextFilesEmbeddingInstanceIterator cnnTextIter = new CnnTextFilesEmbeddingInstanceIterator();
    cnnTextIter.setTrainBatchSize(64);
    cnnTextIter.setWordVectorLocation(DatasetLoader.loadGoogleNewsVectors());
    cnnTextIter.setTextsLocation(DatasetLoader.loadAngerFilesDir());
    clf.setInstanceIterator(cnnTextIter);

    cnnTextIter.initialize();
    final WordVectors wordVectors = cnnTextIter.getWordVectors();
    int vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;

    ConvolutionLayer conv1 = new ConvolutionLayer();
    conv1.setKernelSize(new int[]{3, vectorSize});
    conv1.setNOut(10);
    conv1.setStride(new int[]{1, vectorSize});
    conv1.setConvolutionMode(ConvolutionMode.Same);

    ConvolutionLayer conv2 = new ConvolutionLayer();
    conv2.setKernelSize(new int[]{2, vectorSize});
    conv2.setNOut(10);
    conv2.setStride(new int[]{1, vectorSize});
    conv2.setConvolutionMode(ConvolutionMode.Same);

    GlobalPoolingLayer gpl = new GlobalPoolingLayer();

    OutputLayer out = new OutputLayer();
    out.setLossFn(new LossMSE());
    out.setActivationFunction(new ActivationIdentity());

    clf.setLayers(conv1, conv2, gpl, out);
    clf.setCacheMode(CacheMode.MEMORY);
    final Instances data = DatasetLoader.loadAngerMeta();
    TestUtil.holdout(clf, data);
  }

  @Test
  public void testTextCnnTextFilesClassification() throws Exception {
    CnnTextFilesEmbeddingInstanceIterator cnnTextIter = new CnnTextFilesEmbeddingInstanceIterator();
    cnnTextIter.setTrainBatchSize(64);
    cnnTextIter.setWordVectorLocation(DatasetLoader.loadGoogleNewsVectors());
    cnnTextIter.setTextsLocation(DatasetLoader.loadAngerFilesDir());
    clf.setInstanceIterator(cnnTextIter);

    cnnTextIter.initialize();
    final WordVectors wordVectors = cnnTextIter.getWordVectors();
    int vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;

    ConvolutionLayer conv1 = new ConvolutionLayer();
    conv1.setKernelSize(new int[]{4, vectorSize});
    conv1.setNOut(10);
    conv1.setStride(new int[]{1, vectorSize});
    conv1.setConvolutionMode(ConvolutionMode.Same);
    conv1.setActivationFunction(new ActivationReLU());

    ConvolutionLayer conv2 = new ConvolutionLayer();
    conv2.setKernelSize(new int[]{3, vectorSize});
    conv2.setNOut(10);
    conv2.setStride(new int[]{1, vectorSize});
    conv2.setConvolutionMode(ConvolutionMode.Same);
    conv2.setActivationFunction(new ActivationReLU());

    GlobalPoolingLayer gpl = new GlobalPoolingLayer();

    OutputLayer out = new OutputLayer();

    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    Dropout d = new Dropout();
    d.setP(0.2);
    nnc.setDropout(d);
    clf.setNeuralNetConfiguration(nnc);

    clf.setLayers(conv1, conv2, gpl, out);
    clf.setCacheMode(CacheMode.MEMORY);
    final Instances data = DatasetLoader.loadAngerMetaClassification();
    TestUtil.holdout(clf, data);
  }

  @Test(expected = InvalidNetworkArchitectureException.class)
  public void testTextCnnTextSingleOutputLayer() throws Exception {
    CnnTextEmbeddingInstanceIterator cnnTextIter = new CnnTextEmbeddingInstanceIterator();
    cnnTextIter.setTrainBatchSize(64);
    cnnTextIter.setWordVectorLocation(DatasetLoader.loadGoogleNewsVectors());
    clf.setInstanceIterator(cnnTextIter);

    GlobalPoolingLayer gpl = new GlobalPoolingLayer();
    OutputLayer out = new OutputLayer();

    clf.setLayers(gpl, out);
    clf.setCacheMode(CacheMode.MEMORY);
    final Instances data = DatasetLoader.loadAnger();
    TestUtil.holdout(clf, data);
  }

  @Test
  public void testTextCnnTextSingleConv() throws Exception {
    CnnTextEmbeddingInstanceIterator cnnTextIter = new CnnTextEmbeddingInstanceIterator();
    cnnTextIter.setTrainBatchSize(64);
    cnnTextIter.setWordVectorLocation(DatasetLoader.loadGoogleNewsVectors());
    clf.setInstanceIterator(cnnTextIter);

//    final WordVectors wordVectors = cnnTextIter.getWordVectors();
//    int vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
    int vectorSize = 300;
    ConvolutionLayer conv1 = new ConvolutionLayer();
    conv1.setKernelSize(new int[]{4, vectorSize});
    conv1.setNOut(10);
    conv1.setStride(new int[]{1, vectorSize});
    conv1.setConvolutionMode(ConvolutionMode.Same);
    conv1.setActivationFunction(new ActivationReLU());

    GlobalPoolingLayer gpl = new GlobalPoolingLayer();
    OutputLayer out = new OutputLayer();
    out.setLossFn(new LossMSE());
    out.setActivationFunction(new ActivationIdentity());

    clf.setLayers(conv1, gpl, out);
    clf.setCacheMode(CacheMode.MEMORY);
    final Instances data = DatasetLoader.loadAnger();
    TestUtil.holdout(clf, data);
  }

  @Test
  public void testSetCacheMode() throws Exception {
    clf.setInstanceIterator(idiMnist);

    DenseLayer denseLayer = new DenseLayer();
    denseLayer.setNOut(128);
    denseLayer.setActivationFunction(new ActivationReLU());

    DenseLayer denseLayer2 = new DenseLayer();
    denseLayer2.setNOut(32);
    denseLayer2.setActivationFunction(new ActivationReLU());

    OutputLayer outputLayer = new OutputLayer();
    outputLayer.setActivationFunction(new ActivationSoftmax());
    outputLayer.setLossFn(new LossMCXENT());

    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
    nnc.setSeed(TestUtil.SEED);

    clf.setNeuralNetConfiguration(nnc);
    clf.setLayers(denseLayer, denseLayer2, outputLayer);
    clf.setIterationListener(new EpochListener());

    clf.setCacheMode(CacheMode.FILESYSTEM);
    TestUtil.holdout(clf, dataMnist);
  }

  @Test
  public void testResume() throws Exception {
    clf.setResume(true);
    final ConvolutionInstanceIterator it = new ConvolutionInstanceIterator();
    it.setHeight(1);
    it.setWidth(4);
    clf.setInstanceIterator(it);

    DenseLayer dl = new DenseLayer();
    dl.setNOut(10);

    ConvolutionLayer cl = new ConvolutionLayer();
    cl.setNOut(3);
    cl.setKernelSize(new int[]{1, 1});
    cl.setStride(new int[]{1, 1});

    OutputLayer ol = new OutputLayer();
    clf.setLayers(cl, dl, ol);
    int numEpochs = 5;
    clf.setNumEpochs(numEpochs);
    assertEquals(numEpochs, clf.numEpochs);
    clf.buildClassifier(dataIris);
    assertEquals(numEpochs, clf.numEpochsPerformed);
    assertEquals(numEpochs, clf.numEpochsPerformedThisSession);

    // Save classifier
    String tmpDir = System.getProperty("java.io.tmpdir");
    String clfPath = Paths.get(tmpDir, "dl4j-mlp-clf.ser").toString();
    saveClf(clfPath, clf);

    // Reload classifier and run #numEpochs epochs again
    Dl4jMlpClassifier clfLoaded = readClf(clfPath);
    clfLoaded.buildClassifier(dataIris);

    // Check if epochs are correctly counted
    assertEquals(numEpochs, clfLoaded.numEpochs);
    assertEquals(numEpochs * 2, clfLoaded.numEpochsPerformed);
    assertEquals(numEpochs, clfLoaded.numEpochsPerformedThisSession);

    // Repeat procedure one more time and check again
    saveClf(clfPath, clfLoaded);
    Dl4jMlpClassifier clfLoaded2 = readClf(clfPath);
    clfLoaded2.buildClassifier(dataIris);

    assertEquals(numEpochs, clfLoaded2.numEpochs);
    assertEquals(numEpochs * 3, clfLoaded2.numEpochsPerformed);
    assertEquals(numEpochs, clfLoaded2.numEpochsPerformedThisSession);

    Files.delete(Paths.get(clfPath));
  }

}
