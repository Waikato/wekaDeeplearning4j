package weka.classifiers.functions;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.iterators.ImageDataSetIterator;
import weka.dl4j.layers.*;

import java.io.File;

/**
 * JUnit tests for the Dl4jMlpClassifier.
 * Tests nominal classes with iris, numerical classes with diabetes and image
 * classification with minimal mnist.
 *
 * @author Steven Lang
 * @version $Revision : 11711 $
 */
public class Dl4jMlpTest {
    
    /**
     * Number of classes in the iris dataset
     */
    private static final int NUM_CLASSES_IRIS = 3;
    
    /**
     * Number of classes in the mnist dataset
     */
    private static final int NUM_CLASSES_MNIST = 10;
    
    /**
     * Number of classes in the diabetes dataset
     */
    private static final int NUM_CLASSES_DIABETES = 1;
    
    /**
     * Number of instances in the iris dataset
     */
    private static final int NUM_INSTANCES_IRIS = 150;
    
    /**
     * Number of instances in the mnist dataset
     */
    private static final int NUM_INSTANCES_MNIST = 10;
    
    /**
     * Number of instances in the diabetes dataset
     */
    private static final int NUM_INSTANCES_DIABETES = 43;
    
    
    /**
     * Test single layer iris.
     *
     * @throws Exception the exception
     */
    @Test
    public void testSingleLayerIris() throws Exception {
        // CLF
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);
        
        // Data
        Instances data = loadIris();
        data.setClassIndex(data.numAttributes() - 1);
        
        OutputLayer outputLayer = new OutputLayer();
        outputLayer.setActivationFn(Activation.SOFTMAX.getActivationFunction());
        outputLayer.setWeightInit(WeightInit.XAVIER);
        
        NeuralNetConfiguration nnc = new NeuralNetConfiguration();
        nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        clf.setNeuralNetConfiguration(nnc);
        clf.setLayers(new Layer[]{outputLayer});
        clf.buildClassifier(data);
        final double[][] res = clf.distributionsForInstances(data);
        
        Assert.assertEquals(NUM_INSTANCES_IRIS, res.length);
        Assert.assertEquals(NUM_CLASSES_IRIS, res[0].length);
    }
    
    /**
     * Test diabetes with one hidden layer
     *
     * @throws Exception
     */
    @Test
    public void testDiabetes() throws Exception {
        // CLF
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);
        
        // Data
        Instances data = loadDiabetes();
        data.setClassIndex(data.numAttributes() - 1);
        
        DenseLayer denseLayer = new DenseLayer();
        denseLayer.setNOut(2);
        denseLayer.setActivationFn(Activation.RELU.getActivationFunction());
        denseLayer.setWeightInit(WeightInit.XAVIER);
        
        OutputLayer outputLayer = new OutputLayer();
        outputLayer.setActivationFn(Activation.SOFTMAX.getActivationFunction());
        outputLayer.setWeightInit(WeightInit.XAVIER);
        
        NeuralNetConfiguration nnc = new NeuralNetConfiguration();
        nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        
        clf.setNeuralNetConfiguration(nnc);
        clf.setLayers(new Layer[]{outputLayer});
        clf.buildClassifier(data);
        final double[][] res = clf.distributionsForInstances(data);
        
        Assert.assertEquals(NUM_INSTANCES_DIABETES, res.length);
        Assert.assertEquals(NUM_CLASSES_DIABETES, res[0].length);
    }
    
    /**
     * Test image dataset iterator mnist.
     *
     * @throws Exception IO error.
     */
    @Test
    public void testImageDatasetIteratorMnist() throws Exception {
        
        // Data
        Instances data = loadMnistMinimalMeta();
        data.setClassIndex(data.numAttributes() - 1);
        ImageDataSetIterator imgIter = loadMnistImageIterator();
        
        final int seed = 1;
        for (int batchSize : new int[]{1, 2, 5, 10}) {
            final int actual = countIterations(data, imgIter, seed, batchSize);
            final int expected = data.numInstances() / batchSize;
            Assert.assertEquals(expected, actual);
        }
    }
    
    /**
     * Counts the number of iterations an {@see ImageDataSetIterator}
     *
     * @param data      Instances to iterate
     * @param imgIter   ImageDataSetIterator to be tested
     * @param seed      Seed
     * @param batchsize Size of the batch which is returned
     *                  in {@see DataSetIterator#next}
     * @return Number of iterations
     * @throws Exception
     */
    private int countIterations(Instances data, ImageDataSetIterator imgIter,
                                int seed, int batchsize) throws Exception {
        DataSetIterator it = imgIter.getIterator(data, seed, batchsize);
        int count = 0;
        while (it.hasNext()) {
            count++;
            DataSet dataset = it.next();
        }
        return count;
    }
    
    
    /**
     * Test minimal mnist conv net.
     *
     * @throws Exception IO error.
     */
    @Test
    public void testMinimalMnistConvNet() throws Exception {
        // CLF
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);
        clf.setNumEpochs(1);
        
        // Data
        Instances data = loadMnistMinimalMeta();
        data.setClassIndex(data.numAttributes() - 1);
        ImageDataSetIterator imgIter = loadMnistImageIterator();
        clf.setDataSetIterator(imgIter);
        
        ConvolutionLayer convLayer1 = new ConvolutionLayer();
        convLayer1.setKernelSizeX(3);
        convLayer1.setKernelSizeY(3);
        convLayer1.setStrideX(1);
        convLayer1.setStrideY(1);
        convLayer1.setActivationFn(Activation.RELU.getActivationFunction());
        convLayer1.setWeightInit(WeightInit.XAVIER);
        convLayer1.setNOut(16);
        convLayer1.setLayerName("Conv-layer 1");
        
        SubsamplingLayer poolLayer1 = new SubsamplingLayer();
        poolLayer1.setPoolingType(PoolingType.MAX);
        poolLayer1.setKernelSizeX(2);
        poolLayer1.setKernelSizeY(2);
        poolLayer1.setStrideX(2);
        poolLayer1.setStrideY(2);
        
        ConvolutionLayer convLayer2 = new ConvolutionLayer();
        convLayer2.setKernelSizeX(3);
        convLayer2.setKernelSizeY(3);
        convLayer2.setStrideX(1);
        convLayer2.setStrideY(1);
        convLayer2.setActivationFn(Activation.RELU.getActivationFunction());
        convLayer2.setWeightInit(WeightInit.XAVIER);
        convLayer2.setNOut(32);
        
        SubsamplingLayer poolLayer2 = new SubsamplingLayer();
        poolLayer2.setPoolingType(PoolingType.MAX);
        poolLayer2.setKernelSizeX(2);
        poolLayer2.setKernelSizeY(2);
        poolLayer2.setStrideX(2);
        poolLayer2.setStrideY(2);
        
        ConvolutionLayer convLayer3 = new ConvolutionLayer();
        convLayer3.setKernelSizeX(3);
        convLayer3.setKernelSizeY(3);
        convLayer3.setStrideX(1);
        convLayer3.setStrideY(1);
        convLayer3.setActivationFn(Activation.RELU.getActivationFunction());
        convLayer3.setWeightInit(WeightInit.XAVIER);
        convLayer3.setNOut(48);
        
        SubsamplingLayer poolLayer3 = new SubsamplingLayer();
        poolLayer3.setPoolingType(PoolingType.MAX);
        poolLayer3.setKernelSizeX(2);
        poolLayer3.setKernelSizeY(2);
        poolLayer3.setStrideX(2);
        poolLayer3.setStrideY(2);
        
        
        DenseLayer denseLayer = new DenseLayer();
        denseLayer.setNOut(128);
        denseLayer.setActivationFn(Activation.RELU.getActivationFunction());
        denseLayer.setWeightInit(WeightInit.XAVIER);
        denseLayer.setLayerName("Dense-layer 1");
        
        OutputLayer outputLayer = new OutputLayer();
        outputLayer.setActivationFn(Activation.SOFTMAX.getActivationFunction());
        outputLayer.setWeightInit(WeightInit.XAVIER);
        outputLayer.setLayerName("Output-layer");
        
        NeuralNetConfiguration nnc = new NeuralNetConfiguration();
        nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        
        
        clf.setNeuralNetConfiguration(nnc);
        clf.setLayers(new Layer[]{convLayer1, poolLayer1, convLayer2, poolLayer2, convLayer3, poolLayer3, denseLayer, outputLayer});
        clf.buildClassifier(data);
        final double[][] res = clf.distributionsForInstances(data);
        
        
        Assert.assertEquals(NUM_INSTANCES_MNIST, res.length);
        Assert.assertEquals(NUM_CLASSES_MNIST, res[0].length);
    }
    
    /**
     * Load the mnist minimal dataset with an ImageDataSetIterator
     * @return ImageDataSetIterator
     */
    private ImageDataSetIterator loadMnistImageIterator() {
        final String imagesPath = "datasets/nominal/mnist-minimal";
        ImageDataSetIterator imgIter = new ImageDataSetIterator();
        imgIter.setImagesLocation(new File(imagesPath));
        final int height = 28;
        final int width = 28;
        final int channels = 1;
        imgIter.setHeight(height);
        imgIter.setWidth(width);
        imgIter.setNumChannels(channels);
        imgIter.setTrainBatchSize(1);
        return imgIter;
    }
    
    /**
     * Load the iris arff file
     * @return Iris data as Instances
     * @throws Exception IO error.
     */
    private Instances loadIris() throws Exception {
        DataSource ds = new DataSource("datasets/nominal/iris.arff");
        Instances data = ds.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }
    
    /**
     * Load the diabetes arff file
     * @return Diabetes data as Instances
     * @throws Exception IO error.
     */
    private Instances loadDiabetes() throws Exception {
        DataSource ds = new DataSource("datasets/numeric/diabetes_numeric.arff");
        Instances data = ds.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }
    
    /**
     * Load the mnist minimal meta arff file
     * @return Mnist minimal meta data as Instances
     * @throws Exception IO error.
     */
    private Instances loadMnistMinimalMeta() throws Exception {
        DataSource ds = new DataSource("datasets/nominal/mnist.meta.minimal.arff");
        Instances data = ds.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        return data;
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
        Instances data = loadDiabetes();
        clf.buildClassifier(data);
        final double[][] res = clf.distributionsForInstances(data);
        Assert.assertEquals(NUM_INSTANCES_DIABETES, res.length);
        Assert.assertEquals(NUM_CLASSES_DIABETES, res[0].length);
    }
    
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
        Instances data = loadMnistMinimalMeta();
        data.setClassIndex(data.numAttributes() - 1);
        final ImageDataSetIterator imgIter = loadMnistImageIterator();
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
        
        Assert.assertEquals(NUM_INSTANCES_MNIST, res.length);
        Assert.assertEquals(NUM_CLASSES_MNIST, res[0].length);
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
        Instances data = loadMnistMinimalMeta();
        data.setClassIndex(data.numAttributes() - 1);
        final ImageDataSetIterator imgIter = loadMnistImageIterator();
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
        final double[][] res = clf.distributionsForInstances(data);
        
        Assert.assertEquals(NUM_INSTANCES_MNIST, res.length);
        Assert.assertEquals(NUM_CLASSES_MNIST, res[0].length);
    }
}
