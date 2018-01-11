package weka.classifiers.functions;

import org.deeplearning4j.nn.conf.layers.Layer;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestName;
import org.nd4j.linalg.activations.Activation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.core.Instances;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.layers.DenseLayer;
import weka.dl4j.layers.OutputLayer;
import weka.util.DatasetLoader;

import java.io.IOException;

/**
 * JUnit tests applying the classifier to different arff datasets.
 *
 * @author Steven Lang
 */
public class DatasetTest {

  /** Logger instance */
  private static final Logger logger = LoggerFactory.getLogger(DatasetTest.class);

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
  /** Start time for time measurement */
  private long startTime;

  @Before
  public void before() {
    // Init mlp clf
    clf = new Dl4jMlpClassifier();
    clf.setSeed(SEED);
    clf.setNumEpochs(DEFAULT_NUM_EPOCHS);
    clf.setDebug(false);

    // Init data
    startTime = System.currentTimeMillis();
    //        TestUtil.enableUIServer(clf);
  }

  @After
  public void after() {
    double time = (System.currentTimeMillis() - startTime) / 1000.0;
    logger.info("Testmethod: " + name.getMethodName());
    logger.info("Time: " + time + "s");
  }

  /**
   * Test date class.
   *
   * @throws Exception IO error.
   */
  @Test
  public void testDateClass() throws Exception {
    runClf(DatasetLoader.loadWineDate());
  }
  /**
   * Test numeric class.
   *
   * @throws Exception IO error.
   */
  @Test
  public void testNumericClass() throws Exception {
    runClf(DatasetLoader.loadFishCatch());
  }

  /**
   * Test nominal class.
   *
   * @throws Exception IO error.
   */
  @Test
  public void testNominal() throws Exception {
    runClf(DatasetLoader.loadIris());
  }

  @Test
  public void testMissingValues() throws Exception {
    runClf(DatasetLoader.loadIrisMissingValues());
  }

  private void runClf(Instances data) throws Exception {
    // Data
    DenseLayer denseLayer = new DenseLayer();
    denseLayer.setNOut(32);
    denseLayer.setLayerName("Dense-layer");
    denseLayer.setActivationFn(Activation.RELU.getActivationFunction());

    OutputLayer outputLayer = new OutputLayer();
    outputLayer.setActivationFn(Activation.SOFTMAX.getActivationFunction());
    outputLayer.setLayerName("Output-layer");

    NeuralNetConfiguration nnc = new NeuralNetConfiguration();

    clf.setNumEpochs(DEFAULT_NUM_EPOCHS);
    clf.setNeuralNetConfiguration(nnc);
    clf.setLayers(denseLayer, outputLayer);

    clf.buildClassifier(data);
    clf.distributionsForInstances(data);
  }
}
