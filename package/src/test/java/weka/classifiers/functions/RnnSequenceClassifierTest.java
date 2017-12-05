package weka.classifiers.functions;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestName;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.stemmers.SnowballStemmer;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.activations.ActivationIdentity;
import weka.dl4j.activations.ActivationTanH;
import weka.dl4j.earlystopping.EarlyStopping;
import weka.dl4j.iterators.instance.TextEmbeddingInstanceIterator;
import weka.dl4j.iterators.instance.TextFilesEmbeddingInstanceIterator;
import weka.dl4j.layers.LSTM;
import weka.dl4j.layers.RnnOutputLayer;
import weka.dl4j.listener.EpochListener;
import weka.dl4j.lossfunctions.LossMSE;
import weka.dl4j.text.stopwords.Dl4jAbstractStopwords;
import weka.dl4j.text.stopwords.Dl4jNull;
import weka.dl4j.text.stopwords.Dl4jRainbow;
import weka.dl4j.text.stopwords.Dl4jWordsFromFile;
import weka.dl4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import weka.dl4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import weka.dl4j.text.tokenization.tokenizer.preprocessor.LowCasePreProcessor;
import weka.dl4j.text.tokenization.tokenizer.preprocessor.StemmingPreprocessor;
import weka.dl4j.text.tokenization.tokenizerfactory.CharacterNGramTokenizerFactory;
import weka.dl4j.text.tokenization.tokenizerfactory.TweetNLPTokenizerFactory;
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

  private static TextEmbeddingInstanceIterator tii;
  private long startTime;

  private FileStatsStorage fss;

  /** Initialize the text instance iterator */
  @BeforeClass
  public static void init() throws IOException {
    modelSlim = DatasetLoader.loadGoogleNewsVectors();
    tii = new TextEmbeddingInstanceIterator();
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

    final TextFilesEmbeddingInstanceIterator tfii = new TextFilesEmbeddingInstanceIterator();
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

  @Test
  public void testConfigRotation() throws Exception {
    Map<String, String> failedConfigs = new HashMap<>();

    tii = new TextEmbeddingInstanceIterator();
    tii.setWordVectorLocation(modelSlim);
    data = DatasetLoader.loadAnger();

    // Reduce datasize
    RemovePercentage rp = new RemovePercentage();
    rp.setPercentage(98);
    rp.setInputFormat(data);
    data = Filter.useFilter(data, rp);

    RnnOutputLayer out = new RnnOutputLayer();
    out.setLossFn(new LossMSE());
    out.setActivationFunction(new ActivationIdentity());

    final Dl4jWordsFromFile wff = new Dl4jWordsFromFile();
    wff.setStopwords(new File("src/test/resources/stopwords/english.txt"));
    // Iterate stopwords
    for (Dl4jAbstractStopwords sw : new Dl4jAbstractStopwords[]{new Dl4jRainbow(), new Dl4jNull(),
        wff}){
      tii.setStopwords(sw);

      final StemmingPreprocessor spp = new StemmingPreprocessor();
      spp.setStemmer(new SnowballStemmer());
      // Iterate TokenPreProcess
      for(TokenPreProcess tpp : new TokenPreProcess[]{
          new CommonPreprocessor(),
          new EndingPreProcessor(),
          new LowCasePreProcessor(),
          spp}){
        tii.setTokenPreProcess(tpp);

        // Iterate tokenizer faktory
        for(TokenizerFactory tf : new TokenizerFactory[]{
            new DefaultTokenizerFactory(),
            new CharacterNGramTokenizerFactory(),
            new TweetNLPTokenizerFactory(),
        }){
          tii.setTokenizerFactory(tf);

          // Create clean classifier
          clf = new RnnSequenceClassifier();
          clf.setNumEpochs(1);
          clf.setLayers(new Layer[]{out});
          clf.setInstanceIterator(tii);
          clf.settBPTTforwardLength(3);
          clf.settBPTTbackwardLength(3);

          String conf = "\n - TokenPreProcess: "+tpp.getClass().getSimpleName()+
                  "\n - TokenizerFactory: "+ tf.getClass().getSimpleName()+
                  "\n - StopWords: " + sw.getClass().getSimpleName();
          log.info(conf);
          try{
            clf.buildClassifier(data);
          } catch (Exception e){
            failedConfigs.put(conf, e.toString());
          }
        }
      }
    }

    // Check if anything failed
    if (!failedConfigs.isEmpty()){
      failedConfigs.forEach((s, s2) -> {
        log.error("Config failed:");
        log.error(s);
        log.error("Exception:");
        log.error(s2);
      });
      Assert.fail();
    }
  }

  @Test
  public void testClassIndexAtPosZero() throws Exception {
    data = DatasetLoader.loadAnger();

    RnnOutputLayer out = new RnnOutputLayer();
    out.setLossFn(new LossMSE());
    out.setActivationFn(new ActivationIdentity());
    clf.setLayers(out);

    // Create reversed attribute list
    ArrayList<Attribute> attsReversed = new ArrayList<>();
    for (int i = data.numAttributes() - 1; i >= 0; i--){
      attsReversed.add(data.attribute(i));
    }

    // Create copy with class at pos 0 and text at pos 1
    Instances copy = new Instances("reversed", attsReversed,data.numInstances());
    data.forEach(d -> {
      Instance inst = new DenseInstance(2);
      inst.setDataset(copy);
      inst.setValue(0, d.classValue());
      inst.setValue(1, d.stringValue(0));
      copy.add(inst);
    });

    copy.setClassIndex(0);

    TestUtil.holdout(clf, copy);
  }


//  @Test
//  public void testImdbDl4j() throws Exception {
//
//    int vectorSize = 300; // Size of the word vectors. 300 in the Google News model
//
//    tii = new TextEmbeddingInstanceIterator();
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
