package weka.zoo;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
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
import weka.core.Instances;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.iterators.instance.ResizeImageInstanceIterator;
import weka.dl4j.zoo.*;
import weka.util.DatasetLoader;
import weka.util.TestUtil;


/**
 * JUnit tests for the ModelZoo ({@link weka.zoo}
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
    public void testPureAlexNet() throws Exception {
        org.deeplearning4j.zoo.model.AlexNet m = new org.deeplearning4j.zoo.model.AlexNet(10, 32, 1);

        int[][] inputShapeModel = m.metaData().getInputShape();
        int height = inputShapeModel[0][1];
        int width = inputShapeModel[0][2];

        inputShapeModel[0][0] = 1;
        inputShapeModel[0][1] = height;
        inputShapeModel[0][2] = width;
        System.out.println("width = " + width);
        m.setInputShape(inputShapeModel);

        MultiLayerConfiguration conf = m.conf();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        int SEED = 42;
        int DEFAULT_BATCHSIZE = 1;
        ImageInstanceIterator iii = DatasetLoader.loadMiniMnistImageIterator();
        iii.setTrainBatchSize(DEFAULT_BATCHSIZE);
        iii = new ResizeImageInstanceIterator(iii, width, height);
        Instances data = DatasetLoader.loadMiniMnistMeta();
        Instances[] split = TestUtil.splitTrainTest(data);
        Instances train = split[0];
        Instances test = split[1];
        DataSetIterator trainIt = iii.getIterator(train, SEED, DEFAULT_BATCHSIZE);
        DataSetIterator testIt = iii.getIterator(test, SEED, DEFAULT_BATCHSIZE);
        logger.info("Train model....");
        for (int i = 0; i < 10; i++) {
            logger.info("Epoch " + i);
            model.fit(trainIt);
            Evaluation evaluate = model.evaluate(trainIt);
            System.out.println("evaluate = " + evaluate.stats());
        }


        logger.info("Evaluate model....");
        Evaluation eval = new Evaluation(10); //create an evaluation object with 10 possible classes
        while (testIt.hasNext()) {
            DataSet next = testIt.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }
        logger.info(eval.stats());
        logger.info("****************Example finished********************");

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

//    @Test
    public void testLeNetMnist() throws Exception {
        performZoo(new LeNet());
    }

//    @Test
    public void testAlexNetMnist() throws Exception {
        performZoo(new AlexNet());
    }

//    @Test
    public void testVGG16() throws Exception {
        performZoo(new VGG16());
    }

//    @Test
    public void testVGG19() throws Exception {
        performZoo(new VGG19());
    }

//    @Test
    public void testSimpleCNN() throws Exception {
        performZoo(new SimpleCNN());
    }

    public void performZoo(ZooModel model) throws Exception {
        // CLF
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);

        // Data
        Instances data = DatasetLoader.loadMiniMnistMeta();
        data.setClassIndex(data.numAttributes() - 1);
        ImageInstanceIterator iterator = DatasetLoader.loadMiniMnistImageIterator();
        iterator.setTrainBatchSize(16);
        clf.setInstanceIterator(iterator);
        clf.setZooModel(model);
        clf.setNumEpochs(10);
//        clf.setIterationListener(new EvaluativeListener(iterator.getIterator(data, 42, 1), 1, InvocationType.EPOCH_END));
        TestUtil.holdout(clf, data);
    }
}
