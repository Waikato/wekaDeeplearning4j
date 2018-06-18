package weka.zoo;

import java.util.Arrays;
import lombok.extern.log4j.Log4j2;
import weka.dl4j.zoo.ResNet50;
import org.junit.Test;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.dl4j.earlystopping.EarlyStopping;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.listener.EpochListener;
import weka.dl4j.zoo.*;
import weka.util.DatasetLoader;

import javax.naming.OperationNotSupportedException;
import java.util.ArrayList;

/**
 * JUnit tests for the ModelZoo ({@link weka.zoo}). Mainly checks out whether the initialization of
 * the models work.
 *
 * @author Steven Lang
 */
@Log4j2
public class ModelZooTest {

  @Test
  public void testLeNetMnist() throws Exception {
    buildModel(new LeNet());
  }

  @Test
  public void testAlexNetMnist() throws Exception {
    buildModel(new AlexNet());
  }

  @Test
  public void testVGG16() throws Exception {
    buildModel(new VGG16());
  }

  @Test
  public void testVGG19() throws Exception {
    buildModel(new VGG19());
  }

  @Test
  public void testResNet50() throws Exception {
    buildModel(new ResNet50());
  }

  private void buildModel(ZooModel model) throws Exception {
    // CLF
    Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
    clf.setSeed(1);

    // Data
    Instances data = DatasetLoader.loadMiniMnistMeta();

    ArrayList<Attribute> atts = new ArrayList<>();
    for (int i = 0; i < data.numAttributes(); i++) {
      atts.add(data.attribute(i));
    }
    Instances shrinkedData = new Instances("shrinked", atts, 10);
    shrinkedData.setClassIndex(1);
    for (int i = 0; i < 10; i++) {
      Instance inst = data.get(i);
      inst.setClassValue(i % 10);
      inst.setDataset(shrinkedData);
      shrinkedData.add(inst);
    }

    ImageInstanceIterator iterator = DatasetLoader.loadMiniMnistImageIterator();
    iterator.setTrainBatchSize(10);
    clf.setInstanceIterator(iterator);
    clf.setZooModel(model);
    clf.setNumEpochs(1);
    final EpochListener epochListener = new EpochListener();
    epochListener.setN(1);
    clf.setIterationListener(epochListener);
    clf.setEarlyStopping(new EarlyStopping(5, 0));
    clf.initializeClassifier(shrinkedData);
    clf.buildClassifier(shrinkedData);
  }


  /** Test CustomNet init */
  @Test(expected = UnsupportedOperationException.class)
  public void testCustomNetInit() throws OperationNotSupportedException {
    new CustomNet().init(0, 0, null);
  }
}
