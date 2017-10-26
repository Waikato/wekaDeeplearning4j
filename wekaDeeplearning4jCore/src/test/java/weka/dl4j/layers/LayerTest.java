package weka.dl4j.layers;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.layers.BatchNormalization;
import weka.dl4j.layers.ConvolutionLayer;
import weka.dl4j.layers.OutputLayer;
import weka.dl4j.layers.SubsamplingLayer;
import weka.dl4j.listener.EpochListener;
import weka.util.DatasetLoader;
import weka.util.TestUtil;


/**
 * JUnit tests for different layers {@link weka.dl4j.layers}
 *
 * @author Steven Lang
 */


public class LayerTest {

    /**
     * Test subsampling layer.
     *
     * @throws Exception
     */
    @Test
    public void testSubsamplingLayer() throws Exception {
        // CLF
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);

        // Data
        Instances data = DatasetLoader.loadMiniMnistMeta();
        data.setClassIndex(data.numAttributes() - 1);
        final ImageInstanceIterator imgIter = DatasetLoader.loadMiniMnistImageIterator();
        clf.setDataSetIterator(imgIter);

        SubsamplingLayer pool = new SubsamplingLayer();
        pool.setKernelSizeX(2);
        pool.setKernelSizeY(2);
        pool.setPoolingType(PoolingType.MAX);

        OutputLayer outputLayer = new OutputLayer();
        outputLayer.setActivationFn(Activation.SOFTMAX.getActivationFunction());
        outputLayer.setWeightInit(WeightInit.XAVIER);

        NeuralNetConfiguration nnc = new NeuralNetConfiguration();
        nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);

        clf.setNeuralNetConfiguration(nnc);
        clf.setLayers(new Layer[]{pool, outputLayer});
        clf.buildClassifier(data);
        final double[][] res = clf.distributionsForInstances(data);

        Assert.assertEquals(DatasetLoader.NUM_INSTANCES_MNIST, res.length);
        Assert.assertEquals(DatasetLoader.NUM_CLASSES_MNIST, res[0].length);
    }

    /**
     * Test a no-hidden-layer neural net (i.e. a perceptron)
     * on the numeric diabetes dataset
     *
     * @throws Exception the exception
     */
    @Test
    public void testOutputLayer() throws Exception {
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        OutputLayer out = new OutputLayer();
        clf.setLayers(new Layer[]{out});
        Instances data = DatasetLoader.loadDiabetes();

        final double[][] res = clf.distributionsForInstances(data);
        clf.buildClassifier(data);
        Assert.assertEquals(DatasetLoader.NUM_INSTANCES_DIABETES, res.length);
        Assert.assertEquals(DatasetLoader.NUM_CLASSES_DIABETES, res[0].length);
    }


    /**
     * Test batchnorm layer.
     *
     * @throws Exception
     */
    @Test
    public void testBatchNormLayer() throws Exception {
        // CLF
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);

        // Data
        Instances data = DatasetLoader.loadMiniMnistMeta();
        data.setClassIndex(data.numAttributes() - 1);
        final ImageInstanceIterator imgIter = DatasetLoader.loadMiniMnistImageIterator();
        clf.setDataSetIterator(imgIter);

        BatchNormalization bn = new BatchNormalization();

        OutputLayer outputLayer = new OutputLayer();
        outputLayer.setActivationFn(Activation.SOFTMAX.getActivationFunction());
        outputLayer.setWeightInit(WeightInit.XAVIER);

        NeuralNetConfiguration nnc = new NeuralNetConfiguration();
        nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);

        clf.setNeuralNetConfiguration(nnc);
        clf.setLayers(new Layer[]{bn, outputLayer});

        clf.buildClassifier(data);
        double[][] res = clf.distributionsForInstances(data);
        Assert.assertEquals(DatasetLoader.NUM_INSTANCES_MNIST, res.length);
        Assert.assertEquals(DatasetLoader.NUM_CLASSES_MNIST, res[0].length);
    }
    /**
     * Test batchnorm layer.
     *
     * @throws Exception
     */
    @Test
    public void testConvolutionalLayer() throws Exception {
        // CLF
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);
        clf.setNumEpochs(10);

        // Data
        Instances data = DatasetLoader.loadMiniMnistMeta();
        final ImageInstanceIterator imgIter = DatasetLoader.loadMiniMnistImageIterator();
        clf.setDataSetIterator(imgIter);

        ConvolutionLayer convLayer = new ConvolutionLayer();
        convLayer.setKernelSize(new int[]{3,3});
        convLayer.setActivationFn(Activation.RELU.getActivationFunction());
        convLayer.setWeightInit(WeightInit.XAVIER);
        convLayer.setNOut(32);

        OutputLayer outputLayer = new OutputLayer();
        outputLayer.setActivationFn(Activation.SOFTMAX.getActivationFunction());
        outputLayer.setWeightInit(WeightInit.XAVIER);

        NeuralNetConfiguration nnc = new NeuralNetConfiguration();
        nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);

        clf.setNeuralNetConfiguration(nnc);
        clf.setLayers(new Layer[]{convLayer, outputLayer});

        clf.buildClassifier(data);
        double[][] res = clf.distributionsForInstances(data);
        Assert.assertEquals(DatasetLoader.NUM_INSTANCES_MNIST, res.length);
        Assert.assertEquals(DatasetLoader.NUM_CLASSES_MNIST, res[0].length);
    }
}
