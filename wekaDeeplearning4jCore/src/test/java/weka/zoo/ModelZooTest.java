package weka.zoo;

import org.junit.Test;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.Instances;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.util.DatasetLoader;
import weka.util.TestUtil;


/**
 * JUnit tests for the ModelZoo ({@link weka.zoo}
 *
 * @author Steven Lang
 */


public class ModelZooTest {


    @Test
    public void testLeNetMnist() throws Exception {
        // CLF
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);

        // Data
        Instances data = DatasetLoader.loadMiniMnistMeta();
        data.setClassIndex(data.numAttributes() - 1);
        ImageInstanceIterator iterator = DatasetLoader.loadMiniMnistImageIterator();
        iterator.setTrainBatchSize(16);
        clf.setDataSetIterator(iterator);
        clf.setZooModel(new weka.dl4j.zoo.LeNet());
        clf.setNumEpochs(10);
        TestUtil.holdout(clf, data);
    }
}
