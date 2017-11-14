package weka.zoo;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.dl4j.earlystopping.EarlyStopping;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.zoo.*;
import weka.util.DatasetLoader;

import java.util.ArrayList;


/**
 * JUnit tests for the ModelZoo ({@link weka.zoo}). Mainly checks out whether the
 * initialization of the models work.
 *
 * @author Steven Lang
 */

@Slf4j
public class ModelZooTest {

    @Test
    public void testLeNetMnist() throws Exception {
        initModel(new LeNet());
    }

    @Test
    public void testAlexNetMnist() throws Exception {
        initModel(new AlexNet());
    }

    @Test
    public void testVGG16() throws Exception {
        initModel(new VGG16());
    }

    @Test
    public void testVGG19() throws Exception {
        initModel(new VGG19());
    }

    @Test
    public void testFaceNetNN4Small2() throws Exception {
        initModel(new FaceNetNN4Small2());
    }

    @Test
    public void testGoogLeNet() throws Exception {
        initModel(new GoogLeNet());
    }

    @Test
    public void testInceptionResNetV1() throws Exception {
        initModel(new InceptionResNetV1());
    }

    @Test
    public void testResNet50() throws Exception {
        initModel(new ResNet50());
    }

    public void initModel(ZooModel model) throws Exception {
        // CLF
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);

        // Data
        Instances data = DatasetLoader.loadMiniMnistMeta();

        ArrayList<Attribute> atts = new ArrayList<>();
        for (int i = 0; i < data.numAttributes(); i++){
            atts.add(data.attribute(i));
        }
        Instances shrinkedData = new Instances("shrinked", atts, 10);
        shrinkedData.setClassIndex(1);
        for (int i = 0; i < 10; i++){
            Instance inst = data.get(i);
            inst.setClassValue(i%10);
            inst.setDataset(shrinkedData);
            shrinkedData.add(inst);
        }

        ImageInstanceIterator iterator = DatasetLoader.loadMiniMnistImageIterator();
        iterator.setTrainBatchSize(10);
        clf.setInstanceIterator(iterator);
        clf.setZooModel(model);
        clf.setNumEpochs(1);
        clf.setEarlyStopping(new EarlyStopping(5, 0));
        clf.initializeClassifier(shrinkedData);
    }

//    @Test
    public void runLeNet() throws Exception {
        // CLF
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);

        // Data
        Instances data = DatasetLoader.loadMiniMnistMeta();

        ImageInstanceIterator iterator = DatasetLoader.loadMiniMnistImageIterator();
        iterator.setTrainBatchSize(32);
        clf.setInstanceIterator(iterator);
        clf.setZooModel(new LeNet());
        clf.setNumEpochs(10);
        clf.setEarlyStopping(new EarlyStopping(5, 0));
        clf.buildClassifier(data);
    }
}
