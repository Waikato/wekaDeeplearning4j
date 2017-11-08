package weka.zoo;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.dl4j.earlystopping.EarlyStopping;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.iterators.instance.ResizeImageInstanceIterator;
import weka.dl4j.zoo.*;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.util.DatasetLoader;
import weka.util.TestUtil;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;


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
        performZoo(new LeNet());
    }

    @Test
    public void testAlexNetMnist() throws Exception {
        performZoo(new AlexNet());
    }

    @Test
    public void testVGG16() throws Exception {
        performZoo(new VGG16());
    }

    @Test
    public void testVGG19() throws Exception {
        performZoo(new VGG19());
    }

    @Test
    public void testFaceNetNN4Small2() throws Exception {
        performZoo(new FaceNetNN4Small2());
    }

    @Test
    public void testGoogLeNet() throws Exception {
        performZoo(new GoogLeNet());
    }

    @Test
    public void testInceptionResNetV1() throws Exception {
        performZoo(new InceptionResNetV1());
    }

    @Test
    public void testResNet50() throws Exception {
        performZoo(new ResNet50());
    }

    public void performZoo(ZooModel model) throws Exception {
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
        clf.setEarlyStoppingConfiguration(new EarlyStopping(5, 0));
        clf.buildClassifier(shrinkedData);
        clf.distributionsForInstances(shrinkedData);
    }
}
