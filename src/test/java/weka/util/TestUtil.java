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
 * TestUtil.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.util;

// import org.deeplearning4j.ui.api.UIServer;
// import org.deeplearning4j.ui.stats.StatsListener;
// import org.deeplearning4j.ui.storage.FileStatsStorage;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TestInstances;
import weka.dl4j.iterators.instance.AbstractInstanceIterator;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

/**
 * Utility class for evaluating classifier in JUnit tests
 *
 * @author Steven Lang
 */
public class TestUtil {

  /**
   * Default number of epochs
   */
  public static final int DEFAULT_NUM_EPOCHS = 1;
  /**
   * Seed
   */
  public static final int SEED = 42;
  /**
   * Default batch size
   */
  public static final int DEFAULT_BATCHSIZE = 32;
  /**
   * Logger instance
   */
  private static final Logger logger = LoggerFactory.getLogger(TestUtil.class);

  /**
   * Perform simple holdout with a given percentage
   *
   * @param clf Classifier
   * @param data Full dataset
   * @param p Split percentage
   */
  public static void holdout(Classifier clf, Instances data, double p) throws Exception {
    Instances[] split = splitTrainTest(data, p);

    Instances train = split[0];
    Instances test = split[1];

    logger.info("Classifier: \n{}", clf.toString());
    clf.buildClassifier(train);
    Evaluation trainEval = new Evaluation(train);
    trainEval.evaluateModel(clf, train);
    logger.info("Weka Train Evaluation:");
    logger.info(trainEval.toSummaryString());
    if (!data.classAttribute().isNumeric()) {
      logger.info(trainEval.toMatrixString());
    }

    Evaluation testEval = new Evaluation(train);
    logger.info("Weka Test Evaluation:");
    testEval.evaluateModel(clf, test);
    logger.info(testEval.toSummaryString());
    if (!data.classAttribute().isNumeric()) {
      logger.info(testEval.toMatrixString());
    }
  }

  /**
   * Perform simple holdout with a given percentage
   *
   * @param clf Classifier
   * @param data Full dataset
   * @param p Split percentage
   */
  public static void holdout(
      Dl4jMlpClassifier clf, Instances data, double p, AbstractInstanceIterator aii)
      throws Exception {

    holdout(clf, data, p);
    Instances[] split = splitTrainTest(data, p);

    Instances test = split[1];
    final DataSetIterator testIter = aii.getDataSetIterator(test, 42);
    final ComputationGraph model = clf.getModel();
    logger.info("DL4J Evaluation: ");
    org.nd4j.evaluation.classification.Evaluation evaluation = model.evaluate(testIter);
    logger.info(evaluation.stats());
  }

  /**
   * Perform simple holdout (2/3, 1/3 split)
   *
   * @param clf Classifier
   * @param data Full datase
   */
  public static void holdout(Classifier clf, Instances data) throws Exception {
    holdout(clf, data, 66);
  }

  /**
   * Perform crossvalidation
   *
   * @param clf Classifier
   * @param data Full dataset
   */
  public static void crossValidate(Classifier clf, Instances data) throws Exception {
    Evaluation ev = new Evaluation(data);
    ev.crossValidateModel(clf, data, 10, new Random(42));
    logger.info(ev.toSummaryString());
  }

  /**
   * Split the dataset into p% train and (100-p)% testImdb set
   *
   * @param data Input data
   * @param p train percentage
   * @return Array of instances: (0) Train, (1) Test
   * @throws Exception Filterapplication went wrong
   */
  public static Instances[] splitTrainTest(Instances data, double p) throws Exception {

    Randomize rand = new Randomize();
    rand.setInputFormat(data);
    rand.setRandomSeed(42);
    data = Filter.useFilter(data, rand);

    RemovePercentage rp = new RemovePercentage();
    rp.setInputFormat(data);
    rp.setPercentage(p);
    rp.setInvertSelection(true);
    Instances train = Filter.useFilter(data, rp);

    rp = new RemovePercentage();
    rp.setInputFormat(data);
    rp.setPercentage(p);
    Instances test = Filter.useFilter(data, rp);

    return new Instances[]{train, test};
  }

  /**
   * Split the dataset into 67% traind an 33% testImdb set
   *
   * @param data Input data
   * @return Array of instances: (0) Train, (1) Test
   * @throws Exception Filterapplication went wrong
   */
  public static Instances[] splitTrainTest(Instances data) throws Exception {
    return splitTrainTest(data, 80);
  }

  /**
   * Convert the classifier to commandline arguments
   *
   * @param clf Classifier
   * @return CLF-String formatted as commandline argument
   */
  public static String toCmdLineArgs(Dl4jMlpClassifier clf) {
    String[] opts = clf.getOptions();
    String res = "";
    for (int i = 0; i < opts.length; i += 2) {
      if (opts[i + 1].equals("NaN")) {
        continue;
      }
      res += opts[i] + " \"" + opts[i + 1].replace("\"", "\\\"") + "\" \\\n";
    }
    return res;
  }

//  /**
//   * Enables the UIServer at http://localhost:9000/train
//   *
//   * @param clf Dl4jMlpClassifier instance
//   */
//  public static void enableUiServer(Dl4jMlpClassifier clf) {
//    // Configure where the network information (gradients, score vs. time etc) is to be
//    //   stored. Here: store in memory.
//    final String tmpfile = "/tmp/out.bin";
//    File f = new File(tmpfile);
//    FileStatsStorage fss = new FileStatsStorage(f);
//    startUiServer(fss);
//    addStatsListener(clf, fss);
//  }
//
//  public static void addStatsListener(Dl4jMlpClassifier clf, FileStatsStorage statsStorage) {
//    clf.setIterationListener(new StatsListener(statsStorage));
//  }
//
//  public static void startUiServer(FileStatsStorage statsStorage) {
//    UIServer uiServer = UIServer.getInstance();
//    uiServer.attach(statsStorage);
//  }

  public static Map<String, Long> getAttributesPerLayer() {
    Map<String, Long> attributesPerLayer = new LinkedHashMap<>();
    attributesPerLayer.put("layer1", 256L);
    attributesPerLayer.put("layer2", 64L);
    return attributesPerLayer;
  }

  public static INDArray get2DArray() {
    double[] row1 = new double[] {-1, -1, -2, 0 };
    double[] row2 = new double[] {2, 2, 4, 5};
    return Nd4j.create(new double[][] {row1, row2});
  }

  public static INDArray get4dActivations() {
    // bs, channels, w, h
    int[] shape = new int[] {16, 512, 64, 64};
    return Nd4j.zeros(shape).add(5);
  }
  /**
   * Creates a test dataset
   */
  public static Instances makeTestDataset(
      int seed,
      int numInstances,
      int numNominal,
      int numNumeric,
      int numString,
      int numDate,
      int numRelational,
      int numClasses,
      int classType,
      int classIndex,
      boolean multiInstance)
      throws Exception {

    TestInstances testset = new TestInstances();
    testset.setSeed(seed);
    testset.setNumInstances(numInstances);
    testset.setNumNominal(numNominal);
    testset.setNumNumeric(numNumeric);
    testset.setNumString(numString);
    testset.setNumDate(numDate);
    testset.setNumRelational(numRelational);
    testset.setNumClasses(numClasses);
    testset.setClassType(classType);
    testset.setClassIndex(classIndex);
    testset.setNumClasses(numClasses);
    testset.setMultiInstance(multiInstance);
    if (numRelational > 0) {
      testset.setNumRelationalNominal(2);
      testset.setNumRelationalString(2);
      testset.setNumRelationalNumeric(2);
      testset.setNumInstancesRelational(75);
    }

    return testset.generate();
  }

  /**
   * Creates a relational test dataset
   */
  public static Instances makeTestDatasetRelational(
      int seed,
      int numInstances,
      int numClasses,
      int classType,
      int classIndex,
      int numRelationalNominal,
      int numRelationalString,
      int numRelationalNumeric,
      int numInstancesRelational)
      throws Exception {

    TestInstances testset = new TestInstances();
    testset.setSeed(seed);
    testset.setNumInstances(numInstances);
    testset.setNumClasses(numClasses);
    testset.setClassType(classType);
    testset.setClassIndex(classIndex);
    testset.setNumClasses(numClasses);
    testset.setMultiInstance(false);

    testset.setNumNominal(0);
    testset.setNumNumeric(0);
    testset.setNumString(0);
    testset.setNumDate(0);
    testset.setNumRelational(1);

    testset.setNumRelationalNominal(numRelationalNominal);
    testset.setNumRelationalString(numRelationalString);
    testset.setNumRelationalNumeric(numRelationalNumeric);
    testset.setNumInstancesRelational(numInstancesRelational);

    final Instances generated = testset.generate();

    // Remove random instances
    Random rand = new Random(42);
    for (Instance datum : generated) {
      final Instances rel = datum.relationalValue(0);
      RemovePercentage rp = new RemovePercentage();
      rp.setInputFormat(rel);
      rp.setPercentage(rand.nextDouble() * 100);
      final Instances rel2 = Filter.useFilter(rel, rp);
      final int i = generated.attribute(0).addRelation(rel2);
      datum.setValue(0, i);
    }
    return generated;
  }


  public static void saveClf(String path, Classifier clf) throws IOException {
    ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path));
    oos.writeObject(clf);
    oos.close();
  }

  public static Dl4jMlpClassifier readClf(String path) throws IOException, ClassNotFoundException {
    ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path));
    Dl4jMlpClassifier clf = (Dl4jMlpClassifier) ois.readObject();
    ois.close();
    return clf;
  }
}
