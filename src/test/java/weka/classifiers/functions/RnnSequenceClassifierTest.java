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
 * RnnSequenceClassifierTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.classifiers.functions;

import static org.junit.Assert.assertEquals;
import static weka.util.TestUtil.readClf;
import static weka.util.TestUtil.saveClf;
import static weka.util.TestUtil.splitTrainTest;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;
import lombok.extern.log4j.Log4j2;
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
import weka.dl4j.enums.GradientNormalization;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.activations.ActivationIdentity;
import weka.dl4j.activations.ActivationTanH;
import weka.dl4j.iterators.instance.sequence.RelationalInstanceIterator;
import weka.dl4j.iterators.instance.sequence.text.rnn.RnnTextEmbeddingInstanceIterator;
import weka.dl4j.iterators.instance.sequence.text.rnn.RnnTextFilesEmbeddingInstanceIterator;
import weka.dl4j.layers.LSTM;
import weka.dl4j.layers.RnnOutputLayer;
import weka.dl4j.lossfunctions.LossMSE;
import weka.dl4j.text.stopwords.Dl4jAbstractStopwords;
import weka.dl4j.text.stopwords.Dl4jNull;
import weka.dl4j.text.stopwords.Dl4jRainbow;
import weka.dl4j.text.stopwords.Dl4jWordsFromFile;
import weka.dl4j.text.tokenization.preprocessor.CommonPreProcessor;
import weka.dl4j.text.tokenization.preprocessor.EndingPreProcessor;
import weka.dl4j.text.tokenization.preprocessor.LowCasePreProcessor;
import weka.dl4j.text.tokenization.preprocessor.StemmingPreProcessor;
import weka.dl4j.text.tokenization.preprocessor.TokenPreProcess;
import weka.dl4j.text.tokenization.tokenizer.factory.CharacterNGramTokenizerFactory;
import weka.dl4j.text.tokenization.tokenizer.factory.DefaultTokenizerFactory;
import weka.dl4j.text.tokenization.tokenizer.factory.TokenizerFactory;
import weka.dl4j.text.tokenization.tokenizer.factory.TweetNLPTokenizerFactory;
import weka.dl4j.updater.Adam;
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
@Log4j2
public class RnnSequenceClassifierTest {

  private static final int batchSize = 64;
  private static final int epochs = 2;
  private static final int truncateLength = 10;
  /**
   * Model path slim
   */
  private static File modelSlim;
  private static RnnTextEmbeddingInstanceIterator tii;
  /**
   * Current name
   */
  @Rule
  public TestName name = new TestName();
  /**
   * Classifier
   */
  private RnnSequenceClassifier clf;
  /**
   * Dataset reuters
   */
  private Instances data;
  private long startTime;

//  private FileStatsStorage fss;

  /**
   * Initialize the text instance iterator
   */
  @BeforeClass
  public static void init() throws IOException {
    modelSlim = DatasetLoader.loadGoogleNewsVectors();
    tii = new RnnTextEmbeddingInstanceIterator();
    tii.setWordVectorLocation(modelSlim);
    tii.setTruncateLength(truncateLength);
    tii.setTrainBatchSize(batchSize);
  }

  @Before
  public void before() {

    // Init mlp clf
    clf = new RnnSequenceClassifier();
    clf.setSeed(TestUtil.SEED);
    clf.setDebug(false);
    clf.setNumEpochs(epochs);

    clf.setInstanceIterator(tii);
    startTime = System.currentTimeMillis();
    //    setupUi();
  }

//  private void setupUi() {
//    String dir = System.getenv("WEKA_HOME");
//    if (dir == null) {
//      dir = Paths.get(System.getenv("HOME"), "wekafiles").toAbsolutePath().toString();
//    }
//    final File dir1 = Paths.get(dir, "network-logs").toAbsolutePath().toFile();
//    dir1.mkdirs();
//    final String f =
//        Paths.get(dir1.toString(), name.getMethodName() + ".out").toAbsolutePath().toString();
//    final File f1 = new File(f);
//    f1.delete();
//    fss = new FileStatsStorage(f1);
//    TestUtil.startUiServer(fss);
//  }

  @After
  public void after() {
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
    nnc.setGradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue);
    nnc.setGradientNormalizationThreshold(1.0);
    Adam opt = new Adam();
    opt.setLearningRate(0.02);
    nnc.setUpdater(opt);

    // Config classifier
    clf.setLayers(lstm1, rnnOut);
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

    TestUtil.holdout(clf, data, 5, tii);
  }

  @Test
  public void testAngerRegression() throws Exception {
    // Define layers
    LSTM lstm1 = new LSTM();
    lstm1.setNOut(5);
    lstm1.setActivationFunction(new ActivationTanH());

    RnnOutputLayer rnnOut = new RnnOutputLayer();
    rnnOut.setLossFn(new LossMSE());
    rnnOut.setActivationFunction(new ActivationIdentity());

    // Network config
    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setL2(1e-5);
    nnc.setGradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue);
    nnc.setGradientNormalizationThreshold(1.0);
    Adam opt = new Adam();
    opt.setLearningRate(0.02);
    nnc.setUpdater(opt);

    // Config classifier
    clf.setLayers(lstm1, rnnOut);
    clf.setNeuralNetConfiguration(nnc);
    clf.settBPTTbackwardLength(2);
    clf.settBPTTforwardLength(2);
    clf.setNumEpochs(1);

    // Randomize data
    data = DatasetLoader.loadAnger();
    data.randomize(new Random(42));

    TestUtil.holdout(clf, data, 1);
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
    nnc.setGradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue);
    nnc.setGradientNormalizationThreshold(1.0);
    Adam opt = new Adam();
    opt.setLearningRate(0.02);
    nnc.setUpdater(opt);

    final RnnTextFilesEmbeddingInstanceIterator tfii = new RnnTextFilesEmbeddingInstanceIterator();
    tfii.setTextsLocation(DatasetLoader.loadAngerFilesDir());
    tfii.setTruncateLength(80);
    tfii.setTrainBatchSize(64);
    tfii.setWordVectorLocation(modelSlim);
    clf.setInstanceIterator(tfii);

    // Config classifier
    clf.setLayers(lstm1, rnnOut);
    clf.setNeuralNetConfiguration(nnc);
    clf.settBPTTbackwardLength(20);
    clf.settBPTTforwardLength(20);

    data = DatasetLoader.loadAngerMeta();
    // Randomize data
    data.randomize(new Random(42));
    TestUtil.holdout(clf, data, 3);
  }

  @Test
  public void testRelationalDataset() throws Exception {
    data = TestUtil
        .makeTestDatasetRelational(TestUtil.SEED, 1000, 2, Attribute.NOMINAL, 1, 2, 2, 2, 100);
    data.setClassIndex(data.numAttributes() - 1);

    // Define layers
    LSTM lstm1 = new LSTM();
    lstm1.setNOut(32);
    lstm1.setActivationFunction(new ActivationTanH());

    RnnOutputLayer rnnOut = new RnnOutputLayer();
    rnnOut.setLossFn(new LossMSE());

    // Network config
    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setL2(1e-5);
    nnc.setGradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue);
    nnc.setGradientNormalizationThreshold(1.0);
    Adam opt = new Adam();
    opt.setLearningRate(0.02);
    nnc.setUpdater(opt);

    final RelationalInstanceIterator rii = new RelationalInstanceIterator();
    rii.setTruncateLength(80);
    rii.setTrainBatchSize(64);
    rii.setRelationalAttributeIndex(0);
    clf.setInstanceIterator(rii);

    // Config classifier
    clf.setLayers(lstm1, rnnOut);
    clf.setNeuralNetConfiguration(nnc);
    clf.settBPTTbackwardLength(20);
    clf.settBPTTforwardLength(20);

    TestUtil.holdout(clf, data, 3);
  }

  @Test
  public void testConfigRotation() throws Exception {
    Map<String, String> failedConfigs = new HashMap<>();

    tii = new RnnTextEmbeddingInstanceIterator();
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
    for (Dl4jAbstractStopwords sw :
        new Dl4jAbstractStopwords[]{new Dl4jRainbow(), new Dl4jNull(), wff}) {
      tii.setStopwords(sw);

      final StemmingPreProcessor spp = new StemmingPreProcessor();
      spp.setStemmer(new SnowballStemmer());
      // Iterate TokenPreProcess
      for (TokenPreProcess tpp :
          new TokenPreProcess[]{
              new CommonPreProcessor(), new EndingPreProcessor(), new LowCasePreProcessor(), spp
          }) {
        tii.setTokenPreProcess(tpp);

        // Iterate tokenizer faktory
        for (TokenizerFactory tf :
            new TokenizerFactory[]{
                new DefaultTokenizerFactory(),
                new CharacterNGramTokenizerFactory(),
                new TweetNLPTokenizerFactory(),
            }) {
          tii.setTokenizerFactory(tf);

          // Create clean classifier
          clf = new RnnSequenceClassifier();
          clf.setNumEpochs(1);
          clf.setLayers(out);
          clf.setInstanceIterator(tii);
          clf.settBPTTforwardLength(3);
          clf.settBPTTbackwardLength(3);

          String conf =
              "\n - TokenPreProcess: "
                  + tpp.getClass().getSimpleName()
                  + "\n - TokenizerFactory: "
                  + tf.getClass().getSimpleName()
                  + "\n - StopWords: "
                  + sw.getClass().getSimpleName();
          log.info(conf);
          try {
            clf.buildClassifier(data);
          } catch (Exception e) {
            failedConfigs.put(conf, e.toString());
          }
        }
      }
    }

    // Check if anything failed
    if (!failedConfigs.isEmpty()) {

      final String err =
          failedConfigs
              .keySet()
              .stream()
              .map(s -> "Config failed: " + s + "\nException: " + failedConfigs.get(s))
              .collect(Collectors.joining("\n"));

      Assert.fail("Some of the configs failed:\n" + err);
    }
  }

  @Test
  public void testClassIndexAtPosZero() throws Exception {
    data = DatasetLoader.loadAnger();

    RnnOutputLayer out = new RnnOutputLayer();
    out.setLossFn(new LossMSE());
    out.setActivationFunction(new ActivationIdentity());
    clf.setLayers(out);

    // Create reversed attribute list
    ArrayList<Attribute> attsReversed = new ArrayList<>();
    for (int i = data.numAttributes() - 1; i >= 0; i--) {
      attsReversed.add(data.attribute(i));
    }

    // Create copy with class at pos 0 and text at pos 1
    Instances copy = new Instances("reversed", attsReversed, data.numInstances());
    data.forEach(
        d -> {
          Instance inst = new DenseInstance(2);
          inst.setDataset(copy);
          inst.setValue(0, d.classValue());
          inst.setValue(1, d.stringValue(0));
          copy.add(inst);
        });

    copy.setClassIndex(0);

    TestUtil.holdout(clf, copy);
  }

  @Test
  public void testResume() throws Exception {
    // Define layers
    LSTM lstm1 = new LSTM();
    lstm1.setNOut(5);
    lstm1.setActivationFunction(new ActivationTanH());

    RnnOutputLayer rnnOut = new RnnOutputLayer();
    rnnOut.setLossFn(new LossMSE());
    rnnOut.setActivationFunction(new ActivationIdentity());

    // Network config
    NeuralNetConfiguration nnc = new NeuralNetConfiguration();
    nnc.setL2(1e-5);
    nnc.setGradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue);
    nnc.setGradientNormalizationThreshold(1.0);
    Adam opt = new Adam();
    opt.setLearningRate(0.02);
    nnc.setUpdater(opt);

    // Config classifier
    clf.setLayers(lstm1, rnnOut);
    clf.setNeuralNetConfiguration(nnc);
    clf.settBPTTbackwardLength(2);
    clf.settBPTTforwardLength(2);
    int numEpochs = 2;
    clf.setNumEpochs(numEpochs);

    // Randomize data
    data = DatasetLoader.loadAnger();
    data.randomize(new Random(42));

    Instances[] instances = splitTrainTest(data, 5);
    data = instances[0];

    TestUtil.holdout(clf, data);

    assertEquals(numEpochs, clf.numEpochsPerformed);
    assertEquals(numEpochs, clf.numEpochsPerformedThisSession);

    // Save classifier
    String tmpDir = System.getProperty("java.io.tmpdir");
    String clfPath = Paths.get(tmpDir, "lstm-clf.ser").toString();
    saveClf(clfPath, clf);

    // Reload classifier and run #numEpochs epochs again
    Dl4jMlpClassifier clfLoaded = readClf(clfPath);
    clfLoaded.buildClassifier(data);

    // Check if epochs are correctly counted
    assertEquals(numEpochs, clfLoaded.numEpochs);
    assertEquals(numEpochs * 2, clfLoaded.numEpochsPerformed);
    assertEquals(numEpochs, clfLoaded.numEpochsPerformedThisSession);

    // Repeat procedure one more time and check again
    saveClf(clfPath, clfLoaded);
    Dl4jMlpClassifier clfLoaded2 = readClf(clfPath);
    clfLoaded2.buildClassifier(data);

    assertEquals(numEpochs, clfLoaded2.numEpochs);
    assertEquals(numEpochs * 3, clfLoaded2.numEpochsPerformed);
    assertEquals(numEpochs, clfLoaded2.numEpochsPerformedThisSession);

    Files.delete(Paths.get(clfPath));
  }
}
