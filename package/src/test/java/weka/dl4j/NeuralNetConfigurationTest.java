package weka.dl4j;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.*;
import org.junit.rules.TestName;
import org.nd4j.linalg.learning.config.IUpdater;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.Instances;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.layers.BatchNormalization;
import weka.dl4j.layers.ConvolutionLayer;
import weka.dl4j.layers.DenseLayer;
import weka.dl4j.layers.OutputLayer;
import weka.dl4j.updater.*;
import weka.util.DatasetLoader;

import java.io.*;
import java.nio.file.Paths;

/**
 * JUnit tests for the NeuralNetConfiguration. Tests setting parameters and different
 * configurations.
 *
 * @author Steven Lang
 */
public class NeuralNetConfigurationTest {

  /** Logger instance */
  private static final Logger logger = LoggerFactory.getLogger(NeuralNetConfigurationTest.class);

  /** Default number of epochs */
  private static final int DEFAULT_NUM_EPOCHS = 1;

  /** Seed */
  private static final int SEED = 42;

  /** Default batch size */
  private static final int DEFAULT_BATCHSIZE = 32;
  /** Current name */
  @Rule public TestName name = new TestName();
  /** Classifier */
  private Dl4jMlpClassifier clf;
  /** Dataset mnist */
  private Instances dataMnist;
  /** Mnist image loader */
  private ImageInstanceIterator idiMnist;
  /** Start time for time measurement */
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

    //        logger.info("Press anything to close");
    //        Scanner sc = new Scanner(System.in);
    //        sc.next();
    double time = (System.currentTimeMillis() - startTime) / 1000.0;
    logger.info("Testmethod: " + name.getMethodName());
    logger.info("Time: " + time + "s");
  }

  /**
   * Test
   *
   * @throws Exception
   */
  @Test
  public void testNNCParameters() throws Exception {
    weka.dl4j.updater.Updater[] updater =
        new weka.dl4j.updater.Updater[] {
          new AdaDelta(),
          new AdaGrad(),
          new Adam(),
          new AdaMax(),
          new Nadam(),
          new Nesterovs(),
          new NoOp(),
          new RmsProp(),
          new Sgd()
        };
    for (weka.dl4j.updater.Updater u : updater) {
      // Define some architecture including all currently available layers
      ConvolutionLayer cl = new ConvolutionLayer();
      cl.setNOut(10);
      DenseLayer dl = new DenseLayer();
      dl.setNOut(10);
      BatchNormalization bn = new BatchNormalization();
      OutputLayer ol = new OutputLayer();
      Layer[] layers = {dl, bn, ol};
      clf.setLayers(layers);

      // Setup the configuration
      NeuralNetConfiguration nnc;

      double l1 = 0.001;
      double l2 = 0.002;
      double lr = 5.0;
      WeightInit weightInit = WeightInit.UNIFORM;

      nnc = new NeuralNetConfiguration();
      nnc.setLearningRate(lr);
      nnc.setUseRegularization(true);
      nnc.setUpdater(u);

      final GradientNormalization clipWiseGN = GradientNormalization.ClipElementWiseAbsoluteValue;
      nnc.setGradientNormalization(clipWiseGN);
      final double gnT1 = 100.0;
      nnc.setGradientNormalizationThreshold(gnT1);

      nnc.setL1(l1);
      nnc.setL2(l2);

      // Not working as of dl4j 0.9.1
      //            nnc.setBiasL1(l1Bias);
      //            nnc.setBiasL2(l2Bias);
      nnc.setWeightInit(weightInit);

      clf.setNeuralNetConfiguration(nnc);
      clf.initializeClassifier(dataMnist); // creates the model internally

      // Get configured layers
      final List<Layer> confLayers = Arrays.stream(clf.getModel().getLayers())
          .map(l -> l.conf().getLayer()).collect(Collectors.toList());
      for (Layer l : confLayers) {
        final BaseLayer bl = (BaseLayer) l;
        IUpdater u2 = bl.getIUpdater();
        double l11 = bl.getL1();
        double l21 = bl.getL2();
        WeightInit weightInit1 = bl.getWeightInit();
        double learningRate = bl.getLearningRate();
        if (!(u instanceof AdaDelta)) { // AdaDelta does not have any learning rate
          Assert.assertEquals(lr, learningRate, 10e-6);
        }

        final GradientNormalization gn = bl.getGradientNormalization();
        final double gnt = bl.getGradientNormalizationThreshold();

        Assert.assertEquals(u.getClass().getSimpleName(), u2.getClass().getSimpleName());
        Assert.assertEquals(l1, l11, 10e-6);
        Assert.assertEquals(l2, l21, 10e-6);
        Assert.assertEquals(weightInit, weightInit1);
        Assert.assertEquals(gnT1, gnt, 10e-5);
        Assert.assertEquals(clipWiseGN, gn);
      }
    }
  }

  @Test
  public void testSerialization() throws IOException, ClassNotFoundException {
    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setSeed(42);
    nnc.builder()
        .learningRate(5)
        .weightInit(WeightInit.UNIFORM)
        .biasLearningRate(5)
        .l1(5)
        .l2(5)
        .updater(new AdaDelta())
        .build();

    final File output = Paths.get(System.getProperty("java.io.tmpdir"), "nnc.object").toFile();
    ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(output));
    oos.writeObject(nnc);
    oos.close();
    ObjectInputStream ois = new ObjectInputStream(new FileInputStream(output));
    NeuralNetConfiguration nnc2 = (NeuralNetConfiguration) ois.readObject();
    Assert.assertEquals(nnc, nnc2);
    output.delete();
  }
}
