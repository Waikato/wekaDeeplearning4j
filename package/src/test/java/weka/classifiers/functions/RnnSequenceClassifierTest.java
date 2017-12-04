package weka.classifiers.functions;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Random;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestName;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import weka.core.Instances;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.activations.ActivationIdentity;
import weka.dl4j.activations.ActivationTanH;
import weka.dl4j.earlystopping.EarlyStopping;
import weka.dl4j.iterators.instance.TextFileInstanceIterator;
import weka.dl4j.iterators.instance.TextInstanceIterator;
import weka.dl4j.layers.LSTM;
import weka.dl4j.layers.RnnOutputLayer;
import weka.dl4j.listener.EpochListener;
import weka.dl4j.lossfunctions.LossMSE;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.util.DatasetLoader;
import weka.util.TestUtil;

/**
 * JUnit tests for the RnnSequenceClassifier. Tests nominal classes with iris, numerical classes
 * with diabetes and image classification with minimal mnist.
 *
 * @author Steven Lang
 */
@Slf4j()
public class RnnSequenceClassifierTest {

  /** Current name */
  @Rule public TestName name = new TestName();
  /** Model path slim */
  private static File modelSlim;
  /** Classifier */
  private RnnSequenceClassifier clf;
  /** Dataset reuters */
  private Instances data;

  private static final int batchSize = 64;
  private static final int epochs = 2;
  private static final int truncateLength = 10;

  private static TextInstanceIterator tii;
  private long startTime;

  private FileStatsStorage fss;

  /** Initialize the text instance iterator */
  @BeforeClass
  public static void init() throws IOException {
    modelSlim = DatasetLoader.loadGoogleNewsVectors();
    tii = new TextInstanceIterator();
    tii.setWordVectorLocation(modelSlim);
    tii.setTruncateLength(truncateLength);
    tii.setTrainBatchSize(batchSize);
  }

  @Before
  public void before() throws Exception {

    // Init mlp clf
    clf = new RnnSequenceClassifier();
    clf.setSeed(TestUtil.SEED);
    clf.setDebug(false);
    clf.setNumEpochs(epochs);

    clf.setInstanceIterator(tii);
    startTime = System.currentTimeMillis();
//    setupUi();
  }

  private void setupUi() {
    String dir = System.getenv("WEKA_HOME");
    if (dir == null) {
      dir = Paths.get(System.getenv("HOME"), "wekafiles").toAbsolutePath().toString();
    }
    final File dir1 = Paths.get(dir, "network-logs").toAbsolutePath().toFile();
    dir1.mkdirs();
    final String f =
        Paths.get(dir1.toString(), name.getMethodName() + ".out").toAbsolutePath().toString();
    final File f1 = new File(f);
    f1.delete();
    fss = new FileStatsStorage(f1);
    TestUtil.startUiServer(fss);
  }

  @After
  public void after() throws IOException {
    double time = (System.currentTimeMillis() - startTime) / 1000.0;
    log.info("Testmethod: " + name.getMethodName());
    log.info("Time: " + time + "s");
  }

  @Test
  public void testImdbClassification() throws Exception {

    // Init data
    data = DatasetLoader.loadImdb();

    // Define layers
    LSTM lstm1 = new LSTM();
    lstm1.setNOut(3);
    lstm1.setActivationFunction(new ActivationTanH());

    RnnOutputLayer rnnOut = new RnnOutputLayer();

    // Network config
    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setL2(1e-5);
    nnc.setUseRegularization(true);
    nnc.setGradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue);
    nnc.setGradientNormalizationThreshold(1.0);
    nnc.setLearningRate(0.02);

    // Config classifier
    clf.setLayers(new Layer[] {lstm1, rnnOut});
    clf.setNeuralNetConfiguration(nnc);
    clf.settBPTTbackwardLength(20);
    clf.settBPTTforwardLength(20);
    clf.setQueueSize(0);

    // Randomize data
    data.randomize(new Random(42));

    // Reduce datasize
    RemovePercentage rp = new RemovePercentage();
    rp.setPercentage(95);
    rp.setInputFormat(data);
    data = Filter.useFilter(data, rp);

    TestUtil.holdout(clf, data, 50, tii);
  }

  @Test
  public void testAngerRegression() throws Exception {
    // Define layers
    LSTM lstm1 = new LSTM();
    lstm1.setNOut(32);
    lstm1.setActivationFunction(new ActivationTanH());

    RnnOutputLayer rnnOut = new RnnOutputLayer();
    rnnOut.setLossFn(new LossMSE());
    rnnOut.setActivationFunction(new ActivationIdentity());

    // Network config
    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setL2(1e-5);
    nnc.setUseRegularization(true);
    nnc.setGradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue);
    nnc.setGradientNormalizationThreshold(1.0);
    nnc.setLearningRate(0.02);

    tii.setTruncateLength(80);
    // Config classifier
    clf.setLayers(new Layer[] {lstm1, rnnOut});
    clf.setNeuralNetConfiguration(nnc);
    clf.settBPTTbackwardLength(20);
    clf.settBPTTforwardLength(20);
//    clf.setQueueSize(4);
    clf.setNumEpochs(3);
    final EpochListener l = new EpochListener();
    l.setN(1);
    clf.setIterationListener(l);
    data = DatasetLoader.loadAnger();
    // Randomize data
    data.randomize(new Random(42));
    TestUtil.holdout(clf, data, 33);
  }



  @Test
  public void testAngerMetaRegression() throws Exception {
    // Define layers
    LSTM lstm1 = new LSTM();
    lstm1.setNOut(32);
    lstm1.setActivationFunction(new ActivationTanH());

    RnnOutputLayer rnnOut = new RnnOutputLayer();
    rnnOut.setLossFn(new LossMSE());
    rnnOut.setActivationFunction(new ActivationIdentity());

    // Network config
    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setL2(1e-5);
    nnc.setUseRegularization(true);
    nnc.setGradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue);
    nnc.setGradientNormalizationThreshold(1.0);
    nnc.setLearningRate(0.02);

    final TextFileInstanceIterator tfii = new TextFileInstanceIterator();
    tfii.setTextsLocation(new File("src/test/resources/numeric/anger-texts"));
    tfii.setTruncateLength(80);
    tfii.setTrainBatchSize(64);
    tfii.setWordVectorLocation(modelSlim);
    clf.setInstanceIterator(tfii);


    // Config classifier
    clf.setLayers(new Layer[] {lstm1, rnnOut});
    clf.setNeuralNetConfiguration(nnc);
    clf.settBPTTbackwardLength(20);
    clf.settBPTTforwardLength(20);
    clf.setNumEpochs(3);

    final EpochListener l = new EpochListener();
    l.setN(1);
    clf.setIterationListener(l);
    clf.setEarlyStopping(new EarlyStopping(5, 10));
    data = DatasetLoader.loadArff("src/test/resources/numeric/anger.meta.arff");
    // Randomize data
    data.randomize(new Random(42));
    TestUtil.holdout(clf, data, 33);
  }



//  @Test
//  public void testImdbDl4j() throws Exception {
//
//    int vectorSize = 300; // Size of the word vectors. 300 in the Google News model
//
//    tii = new TextInstanceIterator();
//    tii.setTruncateLength(truncateLength);
//    tii.setWordVectorLocation(modelSlim);
//    final int bs = batchSize;
//    tii.setTrainBatchSize(bs);
//
//    final Instances[] insts = TestUtil.splitTrainTest(data, 50);
//    Instances trainData = insts[0];
//    Instances testData = insts[1];
//
//    final int seed = 42;
//    DataSetIterator trainIter = tii.getDataSetIterator(trainData, seed, bs);
//    DataSetIterator testIter = tii.getDataSetIterator(testData, seed, bs);
//
//    final int queueSize = 4;
//    trainIter = new AsyncDataSetIterator(trainIter, queueSize);
//    testIter = new AsyncDataSetIterator(testIter, queueSize);
//
//    // Download and extract data
//    Nd4j.getMemoryManager().togglePeriodicGc(false); // https://deeplearning4j.org/workspaces
//
//    // Set up network configuration
//    final int n = 256;
//    MultiLayerConfiguration conf =
//        new org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder()
//            .updater(
//                Updater
//                    .ADAM) // To configure: .updater(Adam.builder().beta1(0.9).beta2(0.999).build())
//            .regularization(true)
//            .l2(1e-5)
//            .weightInit(WeightInit.XAVIER)
//            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
//            .gradientNormalizationThreshold(1.0)
//            .learningRate(2e-2)
//            .seed(seed)
//            .trainingWorkspaceMode(WorkspaceMode.SEPARATE)
//            .inferenceWorkspaceMode(WorkspaceMode.SEPARATE) // https://deeplearning4j.org/workspaces
//            .list()
//            .layer(
//                0, new LSTM.Builder().nIn(vectorSize).nOut(n).activation(Activation.TANH).build())
//            .layer(
//                1,
//                new org.deeplearning4j.nn.conf.layers.RnnOutputLayer.Builder()
//                    .activation(Activation.SOFTMAX)
//                    .lossFunction(LossFunctions.LossFunction.MCXENT)
//                    .nIn(n)
//                    .nOut(2)
//                    .build())
//            .pretrain(false)
//            .backprop(true)
//            .build();
//
//    MultiLayerNetwork net = new MultiLayerNetwork(conf);
//    net.init();
//    net.setListeners(new StatsListener(fss));
//
//    log.info("Starting training");
//    StopWatch sw = new StopWatch();
//    for (int i = 0; i < epochs; i++) {
//      sw.start();
//      net.fit(trainIter);
//      sw.stop();
//      trainIter.reset();
//      log.info("Epoch " + i + " complete, took {} . Starting evaluation:", sw.toString());
//      sw.reset();
//      // Run evaluation. This is on 25k reviews, so can take some time
//      Evaluation evaluation = net.evaluate(testIter);
//      log.info(evaluation.stats());
//    }
//  }
}
