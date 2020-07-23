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
 * NeuralNetConfigurationTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import lombok.extern.log4j.Log4j2;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitDistribution;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestName;
import org.nd4j.linalg.learning.regularization.L1Regularization;
import org.nd4j.linalg.learning.regularization.L2Regularization;
import org.nd4j.linalg.learning.regularization.Regularization;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.classifiers.functions.dl4j.Utils;
import weka.core.Instances;
import weka.dl4j.distribution.BinomialDistribution;
import weka.dl4j.distribution.ConstantDistribution;
import weka.dl4j.distribution.Distribution;
import weka.dl4j.distribution.LogNormalDistribution;
import weka.dl4j.distribution.NormalDistribution;
import weka.dl4j.distribution.OrthogonalDistribution;
import weka.dl4j.distribution.TruncatedNormalDistribution;
import weka.dl4j.distribution.UniformDistribution;
import weka.dl4j.dropout.AbstractDropout;
import weka.dl4j.dropout.AlphaDropout;
import weka.dl4j.dropout.Dropout;
import weka.dl4j.dropout.GaussianDropout;
import weka.dl4j.dropout.GaussianNoise;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.layers.ConvolutionLayer;
import weka.dl4j.layers.DenseLayer;
import weka.dl4j.layers.OutputLayer;
import weka.dl4j.updater.AdaDelta;
import weka.dl4j.updater.AdaGrad;
import weka.dl4j.updater.AdaMax;
import weka.dl4j.updater.Adam;
import weka.dl4j.updater.Nadam;
import weka.dl4j.updater.Nesterovs;
import weka.dl4j.updater.NoOp;
import weka.dl4j.updater.RmsProp;
import weka.dl4j.updater.Sgd;
import weka.dl4j.updater.Updater;
import weka.dl4j.weightnoise.AbstractWeightNoise;
import weka.dl4j.weightnoise.DropConnect;
import weka.dl4j.weightnoise.WeightNoise;
import weka.util.DatasetLoader;

/**
 * JUnit tests for the NeuralNetConfiguration. Tests setting parameters and different
 * configurations.
 *
 * @author Steven Lang
 */
@Log4j2
public class NeuralNetConfigurationTest {

  /**
   * Default number of epochs
   */
  private static final int DEFAULT_NUM_EPOCHS = 1;

  /**
   * Seed
   */
  private static final int SEED = 42;

  /**
   * Default batch size
   */
  private static final int DEFAULT_BATCHSIZE = 32;
  /**
   * Current name
   */
  @Rule
  public TestName name = new TestName();
  /**
   * Fail message builder
   */
  StringBuilder failMessage = new StringBuilder();
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
   * Start time for time measurement
   */
  private long startTime;

  @Before
  public void before() throws Exception {
    // Init mlp clf
    clf = new Dl4jMlpClassifier();
    clf.setSeed(SEED);
    clf.setNumEpochs(DEFAULT_NUM_EPOCHS);
    clf.setDebug(false);

    // Init data
    dataMnist = DatasetLoader.loadMiniMnistMeta();
    idiMnist = DatasetLoader.loadMiniMnistImageIterator();
    idiMnist.setTrainBatchSize(DEFAULT_BATCHSIZE);
    startTime = System.currentTimeMillis();
    clf.setInstanceIterator(idiMnist);
    //        TestUtil.enableUiServer(clf);
  }

  @After
  public void after() throws IOException {

    //        log.info("Press anything to close");
    //        Scanner sc = new Scanner(System.in);
    //        sc.next();
    double time = (System.currentTimeMillis() - startTime) / 1000.0;
    log.info("Testmethod: " + name.getMethodName());
    log.info("Time: " + time + "s");
  }

  @After
  public void checkFailMessage() {
    String fails = failMessage.toString();
    if (fails.isEmpty()) {
      return;
    }

    String failMessage = "Failed Cases:\n" + fails;
    fail(failMessage);
  }

  @Test
  public void testSerialization() throws IOException, ClassNotFoundException {
    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setSeed(42);
    nnc.setWeightInit(WeightInit.UNIFORM);
    nnc.setL1(5);
    nnc.setL2(5);
    nnc.setUpdater(new AdaMax());
    nnc.setBiasUpdater(new AdaMax());
    nnc.setDropout(new Dropout());
    nnc.setWeightNoise(new DropConnect());
    nnc.setGradientNormalization(GradientNormalization.None);
    nnc.setDist(new weka.dl4j.distribution.ConstantDistribution());

    nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
    final File output = Paths.get(System.getProperty("java.io.tmpdir"), "nnc.object").toFile();
    ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(output));
    oos.writeObject(nnc);
    oos.close();
    ObjectInputStream ois = new ObjectInputStream(new FileInputStream(output));
    NeuralNetConfiguration nnc2 = (NeuralNetConfiguration) ois.readObject();
    assertEquals(nnc.dist, nnc2.dist);
    assertEquals(nnc.dropout, nnc2.dropout);
    assertEquals(nnc.updater, nnc2.updater);
    assertEquals(nnc.biasUpdater, nnc2.biasUpdater);
    assertEquals(nnc.weightNoise, nnc2.weightNoise);
    assertEquals(nnc, nnc2);
    output.delete();
  }

  /**
   * Get all available updaters and initialize them with non default parameters.
   *
   * @return List of Updater
   */
  public List<Updater> getAvailableUpdaterWithNonDefaultParameters() {
    List<Updater> updaters = new ArrayList<>();
    AdaDelta adaDelta = new AdaDelta();
    adaDelta.setEpsilon(10);
    adaDelta.setLearningRate(10);
    adaDelta.setRho(10);
    updaters.add(adaDelta);

    AdaGrad adaGrad = new AdaGrad();
    adaGrad.setEpsilon(10);
    adaGrad.setLearningRate(10);
    updaters.add(adaGrad);

    Adam adam = new Adam();
    adam.setLearningRate(10);
    adam.setBeta1(10);
    adam.setBeta2(10);
    adam.setEpsilon(10);
    updaters.add(adam);

    AdaMax adaMax = new AdaMax();
    adaMax.setBeta1(10);
    adaMax.setBeta2(10);
    adaMax.setEpsilon(10);
    adaMax.setLearningRate(10);
    updaters.add(adaMax);

    Nadam nadam = new Nadam();
    nadam.setBeta1(10);
    nadam.setBeta2(10);
    nadam.setEpsilon(10);
    nadam.setLearningRate(10);
    updaters.add(nadam);

    Nesterovs nesterovs = new Nesterovs();
    nesterovs.setMomentum(10);
    nesterovs.setLearningRate(10);
    updaters.add(nesterovs);

    NoOp noOp = new NoOp();
    updaters.add(noOp);

    RmsProp rmsProp = new RmsProp();
    rmsProp.setEpsilon(10);
    rmsProp.setRmsDecay(10);
    rmsProp.setLearningRate(10);
    updaters.add(rmsProp);

    Sgd sgd = new Sgd();
    sgd.setLearningRate(10);
    updaters.add(sgd);

    return updaters;
  }

  @Test
  public void testGradientNormalizationThreshold() throws Exception {
    for (double gradientNormalizationThreshold : new double[]{0.0, 0.1, 1.0, 10}) {
      NeuralNetConfiguration conf = new NeuralNetConfiguration();
      conf.setGradientNormalizationThreshold(gradientNormalizationThreshold);
      checkAppliedParameters(
          conf, gradientNormalizationThreshold, BaseLayer::getGradientNormalizationThreshold);
    }
  }

  @Test
  public void testGradientNormalization() throws Exception {
    for (GradientNormalization gn : GradientNormalization.values()) {
      NeuralNetConfiguration conf = new NeuralNetConfiguration();
      conf.setGradientNormalization(gn);
      checkAppliedParameters(conf, gn, BaseLayer::getGradientNormalization);
    }
  }

  @Test
  public void testWeightNoise() throws Exception {
    for (AbstractWeightNoise wn :
        new AbstractWeightNoise[]{new DropConnect(), new WeightNoise()}) {
      NeuralNetConfiguration conf = new NeuralNetConfiguration();
      conf.setWeightNoise(wn);
      checkAppliedParameters(conf, wn, BaseLayer::getWeightNoise);
    }
  }

  @Test
  public void testOptimizationAlgo() throws Exception {
    for (OptimizationAlgorithm optAlgo : OptimizationAlgorithm.values()) {
      NeuralNetConfiguration conf = new NeuralNetConfiguration();
      conf.setOptimizationAlgo(optAlgo);
      log.info(optAlgo.toString());
      final Dl4jMlpClassifier clf = setupClf(conf);
      final OptimizationAlgorithm actual = clf.getModel().conf().getOptimizationAlgo();
      if (!actual.equals(optAlgo)) {
        failMessage.append(String.format("actual=%s,expected=%s", actual, optAlgo));
      }
    }
  }

  @Test
  public void testBiasUpdater() throws Exception {
    for (Updater updater : getAvailableUpdaterWithNonDefaultParameters()) {
      NeuralNetConfiguration conf = new NeuralNetConfiguration();
      conf.setBiasUpdater(updater);
      checkAppliedParameters(conf, updater, BaseLayer::getBiasUpdater);
    }
  }

  @Test
  public void testUpdater() throws Exception {
    for (Updater updater : getAvailableUpdaterWithNonDefaultParameters()) {
      NeuralNetConfiguration conf = new NeuralNetConfiguration();
      conf.setUpdater(updater);
      checkAppliedParameters(conf, updater, BaseLayer::getIUpdater);
    }
  }

  @Test
  public void testWeightInit() throws Exception {
    List<WeightInit> skipWeightInits = new ArrayList<>();
    skipWeightInits.add(WeightInit.IDENTITY);
    skipWeightInits.add(WeightInit.DISTRIBUTION);
    for (WeightInit wi : WeightInit.values()) {
      if (skipWeightInits.contains(wi)) {
        continue;
      }
      NeuralNetConfiguration conf = new NeuralNetConfiguration();
      conf.setWeightInit(wi);
      checkAppliedParameters(conf, wi.getWeightInitFunction(), BaseLayer::getWeightInitFn);
    }

  }

  @Test
  public void testDropout() throws Exception {
    for (AbstractDropout dropout :
        new AbstractDropout[]{
            new AlphaDropout(), new Dropout(), new GaussianDropout(), new GaussianNoise()
        }) {
      NeuralNetConfiguration conf = new NeuralNetConfiguration();
      conf.setDropout(dropout);
      checkAppliedParameters(conf, dropout, BaseLayer::getIDropout);
    }
  }

  @Test
  public void testL1() throws Exception {
    for (double l1 : new double[]{0.0, 0.1, 1.0, 10}) {
      NeuralNetConfiguration conf = new NeuralNetConfiguration();
      conf.setL1(l1);

      final List<BaseLayer> layers = getConfiguredLayers(conf);

      for (BaseLayer layer : layers) {
        List<Regularization> regs= layer.getRegularization();

        // If l1 was 0, check that list is empty
        if (l1 < 1e-7){
          if (regs.size() > 0){
            failMessage.append(
                String.format("expected=%s, actual=%s, BaseLayer=%s\n", l1, ((L1Regularization) regs.get(0)).getL1().valueAt(0, 0), layer));
          }
          continue;
        }

        L1Regularization reg = (L1Regularization) regs.get(0);

        double actual = reg.getL1().valueAt(0, 0);
        if (!((actual - l1 < 1e-7))) {
          failMessage.append(
              String.format("expected=%s, actual=%s, BaseLayer=%s\n", l1, actual, layer));
        }
      }
    }
  }

  @Test
  public void testL2() throws Exception {
    for (double l2 : new double[]{0.0, 0.2, 2.0, 20}) {
      NeuralNetConfiguration conf = new NeuralNetConfiguration();
      conf.setL2(l2);
      final List<BaseLayer> layers = getConfiguredLayers(conf);

      for (BaseLayer layer : layers) {
        List<Regularization> regs= layer.getRegularization();

        // If l2 was 0, check that list is empty
        if (l2 < 1e-7){
          if (regs.size() > 0){
            failMessage.append(
                String.format("expected=%s, actual=%s, BaseLayer=%s\n", l2, ((L2Regularization) regs.get(0)).getL2().valueAt(0, 0), layer));
          }
          continue;
        }


        L2Regularization reg = (L2Regularization) regs.get(0);

        double actual = reg.getL2().valueAt(0, 0);
        if (!((actual - l2 < 1e-7))) {
          failMessage.append(
              String.format("expected=%s, actual=%s, BaseLayer=%s\n", l2, actual, layer));
        }
      }
    }
  }

  @Test
  public void testBiasInit() throws Exception {
    for (double biasInit : new double[]{0.0, 1.0, 10}) {
      NeuralNetConfiguration conf = new NeuralNetConfiguration();
      conf.setBiasInit(biasInit);
      checkAppliedParameters(conf, biasInit, BaseLayer::getBiasInit);
    }
  }

// Since 1.0.0-beta5, getDist does not exist anymore
  @Test
  public void testDistribution() throws Exception {
    for (Distribution dist :
        new Distribution[]{
            new ConstantDistribution(),
            new LogNormalDistribution(),
            new OrthogonalDistribution(),
            new TruncatedNormalDistribution(),
            new BinomialDistribution(),
            new NormalDistribution(),
            new UniformDistribution()
        }) {
      NeuralNetConfiguration conf = new NeuralNetConfiguration();
      conf.setDist(dist);
      conf.setWeightInit(WeightInit.DISTRIBUTION);

      final List<BaseLayer> layers = getConfiguredLayers(conf);

      for (BaseLayer layer : layers) {
        WeightInitDistribution actual = (WeightInitDistribution) layer.getWeightInitFn();
        WeightInitDistribution expected = new WeightInitDistribution(dist.getBackend());
        if (!expected.equals(actual)) {
          failMessage.append(
              String.format("expected=%s, actual=%s, BaseLayer=%s\n", expected, actual, layer));
        }
      }


    }
  }

  private void checkAppliedParameters(
      NeuralNetConfiguration conf, Object expected, Function<BaseLayer, Object> getter)
      throws Exception {
    final List<BaseLayer> layers = getConfiguredLayers(conf);

    for (BaseLayer layer : layers) {
      final Object actual = getter.apply(layer);
      if (expected instanceof ApiWrapper) {
        expected = ((ApiWrapper) expected).getBackend();
      }
      if (!expected.equals(actual)) {
        failMessage.append(
            String.format("expected=%s, actual=%s, BaseLayer=%s\n", expected, actual, layer));
      }
    }
  }

  private List<BaseLayer> getConfiguredLayers(NeuralNetConfiguration conf) throws Exception {
    Dl4jMlpClassifier clf = setupClf(conf);

    return Arrays.stream(clf.getModel().getLayers())
        .map(l -> (BaseLayer) l.conf().getLayer())
        .collect(Collectors.toList());
  }

  private Dl4jMlpClassifier setupClf(NeuralNetConfiguration conf) throws Exception {
    Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
    clf.setNeuralNetConfiguration(conf);

    final ConvolutionLayer convolutionLayer = new ConvolutionLayer();
    convolutionLayer.setNOut(2);
    final DenseLayer denseLayer = new DenseLayer();
    denseLayer.setNOut(2);
    final OutputLayer outputLayer = new OutputLayer();
    outputLayer.setNOut(2);
    clf.setLayers(convolutionLayer, denseLayer, outputLayer);

    final Instances data = DatasetLoader.loadMiniMnistMeta();
    final ImageInstanceIterator iii = DatasetLoader.loadMiniMnistImageIterator();
    clf.setInstanceIterator(iii);
    clf.initializeClassifier(data);
    return clf;
  }
}
