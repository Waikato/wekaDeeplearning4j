package weka.zoo;

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


public class ModelZooTest {


    /**
     * Logger instance
     */
    private static final Logger logger = LoggerFactory.getLogger(ModelZooTest.class);

    /**
     * Test AlexNet with resizing Mnist (TODO: Fix)
     *
     * @throws Exception
     */
//    @Test
    public static void testPureLeNetCifar() throws Exception {
        org.deeplearning4j.zoo.model.LeNet m = new org.deeplearning4j.zoo.model.LeNet(10, 32, 1);

        CifarDataSetIterator cdiTrain = new CifarDataSetIterator(256, 40000,
                true);
        CifarDataSetIterator cdiTest = new CifarDataSetIterator(256, 10000,
                false);
        int[][] inputShapeModel = m.metaData().getInputShape();
        inputShapeModel[0][0] = 3;
        inputShapeModel[0][1] = 32;
        inputShapeModel[0][2] = 32;
        m.setInputShape(inputShapeModel);

        MultiLayerConfiguration conf = m.conf();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
//        ComputationGraph model= mlpToCG(conf, inputShapeModel);
//        model.init();

        logger.info("Train model....");
        for (int i = 0; i < 10; i++) {
            logger.info("Epoch " + i);
            long t0 = System.currentTimeMillis();
            model.fit(cdiTrain);
            long t1 = System.currentTimeMillis();
            System.out.println("(t1-t0)/1000 = " + (t1 - t0) / 1000d);
            Evaluation evaluate = model.evaluate(cdiTrain);
            System.out.println("evaluate = " + evaluate.stats());
        }


        logger.info("Evaluate model....");
        Evaluation eval = new Evaluation(10); //create an evaluation object with 10 possible classes
        while (cdiTest.hasNext()) {
            DataSet next = cdiTest.next();
//            INDArray output = model.outputSingle(next.getFeatureMatrix()); //get the networks prediction
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
//            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class

        }
        logger.info(eval.stats());
        logger.info("****************Example finished********************");

    }

    private ComputationGraph mlpToCG(MultiLayerConfiguration mlc, int[][] shape){
        ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder().graphBuilder();
        List<NeuralNetConfiguration> confs = mlc.getConfs();
        String currentInput = "input";
        builder.addInputs(currentInput);
        for (NeuralNetConfiguration conf : confs){
            Layer l = conf.getLayer();
            String layerName = l.getLayerName();
            builder.addLayer(layerName, l, currentInput);
            currentInput = layerName;
        }
        builder.setOutputs(currentInput);
        builder.setInputTypes(InputType.convolutional(shape[0][1], shape[0][2], shape[0][0]));
        ComputationGraphConfiguration cgc = builder.build();
        return new ComputationGraph(cgc);
    }


    //    @Test
    public void testLeNetMnistReshaped() throws Exception {
        // CLF
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);

        // Data
        Instances data = DatasetLoader.loadMiniMnistMeta();
        data.setClassIndex(data.numAttributes() - 1);
        ImageInstanceIterator iterator = DatasetLoader.loadMiniMnistImageIterator();
        iterator.setTrainBatchSize(16);

        clf.setInstanceIterator(new ResizeImageInstanceIterator(iterator, 80, 80));
        clf.setZooModel(new LeNet());
        clf.setNumEpochs(10);
        TestUtil.holdout(clf, data);
    }

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
        for (int i = 0; i < 100; i++){
            Instance inst = data.get(i);
            inst.setClassValue(i%10);
            inst.setDataset(shrinkedData);
            shrinkedData.add(inst);
        }

        ImageInstanceIterator iterator = DatasetLoader.loadMiniMnistImageIterator();
        iterator.setTrainBatchSize(20);
        clf.setInstanceIterator(iterator);
        clf.setZooModel(model);
        clf.setNumEpochs(3);
        clf.buildClassifier(shrinkedData);
        clf.distributionsForInstances(shrinkedData);
    }
}
