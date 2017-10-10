package weka.classifiers.functions;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.iterators.ConvolutionalInstancesIterator;
import weka.dl4j.iterators.ImageDataSetIterator;
import weka.dl4j.layers.ConvolutionLayer;
import weka.dl4j.layers.DenseLayer;
import weka.dl4j.layers.OutputLayer;
import weka.dl4j.layers.SubsamplingLayer;
import weka.dl4j.updater.Sgd;

import java.io.File;
/**
 * JUnit tests for the Dl4jMlpClassifier.
 * Tests nominal classes with iris, numerical classes with diabetes and image
 * classification with minimal mnist.
 *
 * @author Steven Lang
 *
 * @version $Revision: 11711 $
 */
public class Dl4jMlpTest {
    
    @Test
    public void testIris() throws Exception {
        // Paths
        final String arffPath = "../datasets/nominal/iris.arff";
        
        // CLF
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);
        clf.setDebug(true);
        
        // Data
        ConverterUtils.DataSource ds = new ConverterUtils.DataSource(arffPath);
        Instances data = ds.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        
        OutputLayer outputLayer = new OutputLayer();
        outputLayer.setActivationFn(Activation.SOFTMAX.getActivationFunction());
        outputLayer.setWeightInit(WeightInit.XAVIER);
        
        NeuralNetConfiguration nnc = new NeuralNetConfiguration();
        nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        final Sgd iUpdater = new Sgd();
        iUpdater.setLearningRate(0.1);
        nnc.setIUpdater(iUpdater);
        
        clf.setNeuralNetConfiguration(nnc);
        clf.setLayers(new Layer[]{outputLayer});
        clf.buildClassifier(data);
        final double[][] res = clf.distributionsForInstances(data);
        
        Assert.assertEquals(150, res.length);
        Assert.assertEquals(3, res[0].length);
    }
    
    @Test
    public void testDl4jLayers(){
        org.deeplearning4j.nn.conf.layers.DenseLayer dl = new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()
                .learningRate(0.01)
                .updater(new org.nd4j.linalg.learning.config.Sgd(0.001))
                .build();
    }
    
    @Test
    public void testDiabetes() throws Exception {
        // Paths
        final String arffPath = "../datasets/numeric/diabetes_numeric.arff";
        
        // CLF
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);
        
        // Data
        ConverterUtils.DataSource ds = new ConverterUtils.DataSource(arffPath);
        Instances data = ds.getDataSet();
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
        
        Assert.assertEquals(43, res.length);
        Assert.assertEquals(1, res[0].length);
    }
    
    
    @Test
    public void testMinimalMnistConvNet() throws Exception {
        // Paths
        final String arffPath = "../datasets/nominal/mnist.meta.minimal.arff";
        final String imagesPath = "../datasets/nominal/mnist-minimal";
        
        // CLF
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);
        clf.setNumEpochs(1);
        
        // Data
        ConverterUtils.DataSource ds = new ConverterUtils.DataSource(arffPath);
        Instances data = ds.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        ImageDataSetIterator imgIter = new ImageDataSetIterator();
        imgIter.setImagesLocation(new File(imagesPath));
        final int height = 28;
        final int width = 28;
        final int channels = 1;
        imgIter.setHeight(height);
        imgIter.setWidth(width);
        imgIter.setNumChannels(channels);
        imgIter.setTrainBatchSize(1);
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
        nnc.setIUpdater(new Sgd());
        
        clf.setNeuralNetConfiguration(nnc);
        clf.setLayers(new Layer[]{convLayer1, poolLayer1, convLayer2, poolLayer2, convLayer3, poolLayer3, denseLayer, outputLayer});
        clf.buildClassifier(data);
        final double[][] res = clf.distributionsForInstances(data);
        
        
        Assert.assertEquals(10, res.length);
        Assert.assertEquals(10, res[0].length);
    }
}
