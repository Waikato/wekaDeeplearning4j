package weka.classifiers.functions;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.model.LeNet;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.StandardEvaluationMetric;
import weka.core.Instances;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.activations.ActivationReLU;
import weka.dl4j.iterators.DefaultInstancesIterator;
import weka.dl4j.iterators.ImageDataSetIterator;
import weka.dl4j.layers.*;
import weka.dl4j.listener.BatchListener;
import weka.dl4j.lossfunctions.LossMCXENT;
import weka.util.DatasetLoader;
import weka.util.TestUtil;

import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import static weka.util.TestUtil.splitTrainTest;

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
     * Number of epochs per test
     */
    private static final int NUM_EPOCHS = 10;


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
        Instances data = DatasetLoader.loadIris();
        data.setClassIndex(data.numAttributes() - 1);
        
        // Define layers
        OutputLayer outputLayer = new OutputLayer();
        outputLayer.setActivationFn(Activation.SOFTMAX.getActivationFunction());
        outputLayer.setWeightInit(WeightInit.XAVIER);
        outputLayer.setUpdater(Updater.SGD);
        outputLayer.setLearningRate(0.01);
        outputLayer.setBiasLearningRate(0.01);
        outputLayer.setLossFn(new LossMCXENT());
        
        // Configure neural network
        NeuralNetConfiguration nnc = new NeuralNetConfiguration();
        nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        
        clf.setNeuralNetConfiguration(nnc);
        clf.setLayers(new Layer[]{outputLayer});
        clf.setNumEpochs(NUM_EPOCHS);
        clf.getDataSetIterator().setTrainBatchSize(5);

        TestUtil.holdout(clf, data);

        final double[][] res = clf.distributionsForInstances(data);
        
        Assert.assertEquals(DatasetLoader.NUM_INSTANCES_IRIS, res.length);
        Assert.assertEquals(DatasetLoader.NUM_CLASSES_IRIS, res[0].length);
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
        Instances data = DatasetLoader.loadDiabetes();
        data.setClassIndex(data.numAttributes() - 1);
        
        DenseLayer denseLayer = new DenseLayer();
        denseLayer.setNOut(2);
        denseLayer.setActivationFn(Activation.RELU.getActivationFunction());
        denseLayer.setWeightInit(WeightInit.XAVIER);
        
        OutputLayer outputLayer = new OutputLayer();
        outputLayer.setActivationFn(Activation.SOFTMAX.getActivationFunction());
        outputLayer.setWeightInit(WeightInit.XAVIER);
        outputLayer.setLossFn(new LossMCXENT());
        
        NeuralNetConfiguration nnc = new NeuralNetConfiguration();
        nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        
        clf.setNeuralNetConfiguration(nnc);
        clf.setLayers(new Layer[]{outputLayer});

        TestUtil.holdout(clf,data);

        final double[][] res = clf.distributionsForInstances(data);
        
        Assert.assertEquals(DatasetLoader.NUM_INSTANCES_DIABETES, res.length);
        Assert.assertEquals(DatasetLoader.NUM_CLASSES_DIABETES, res[0].length);
    }
    
    /**
     * Test image dataset iterator mnist.
     *
     * @throws Exception IO error.
     */
    @Test
    public void testImageDatasetIteratorMnist() throws Exception {
        
        // Data
        Instances data = DatasetLoader.loadMiniMnistMeta();
        data.setClassIndex(data.numAttributes() - 1);
        ImageDataSetIterator imgIter = DatasetLoader.loadMiniMnistImageIterator();
        
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
        clf.setNumEpochs(NUM_EPOCHS);
        clf.setDebug(false);
        
        // Data
        Instances data = DatasetLoader.loadMediumMnistMeta();
        data.setClassIndex(data.numAttributes() - 1);
        ImageDataSetIterator imgIter = DatasetLoader.loadMediumMnistImageIterator();
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

        TestUtil.holdout(clf, data);

        final double[][] res = clf.distributionsForInstances(data);
        Assert.assertEquals(DatasetLoader.NUM_INSTANCES_MNIST, res.length);
        Assert.assertEquals(DatasetLoader.NUM_CLASSES_MNIST, res[0].length);
    }


    /**
     * Test minimal mnist dense net.
     *
     * @throws Exception IO error.
     */
    @Test
    public void testMinimalMnist() throws Exception {
        // CLF
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(10);
        clf.setNumEpochs(NUM_EPOCHS);

        // Data
        Instances data = DatasetLoader.loadMiniMnistMeta();
        ImageDataSetIterator imgIter = DatasetLoader.loadMiniMnistImageIterator();
        clf.setDataSetIterator(imgIter);

        DenseLayer dl = new DenseLayer();
        dl.setNOut(128);

        ConvolutionLayer cl = new ConvolutionLayer();
        cl.setKernelSize(new int[]{3,3});
        cl.setNOut(16);
        cl.setWeightInit(WeightInit.XAVIER);
        cl.setActivationFn(Activation.RELU.getActivationFunction());

        DenseLayer denseLayer = new DenseLayer();
        denseLayer.setNOut(64);

        OutputLayer outputLayer = new OutputLayer();
        outputLayer.setActivationFn(Activation.SOFTMAX.getActivationFunction());
        outputLayer.setWeightInit(WeightInit.XAVIER);
        outputLayer.setLayerName("Output-layer");

        NeuralNetConfiguration nnc = new NeuralNetConfiguration();
        nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);


        clf.setNeuralNetConfiguration(nnc);
        clf.setLayers(new Layer[]{cl, dl, denseLayer, outputLayer});

        TestUtil.holdout(clf, data);
    }


    /**
     * Test minimal mnist dense net.
     *
     * @throws Exception IO error.
     */
    @Test
    public void testMinimalMnistArff() throws Exception {
        // CLF
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);
        clf.setNumEpochs(10);

        // Data
        Instances data = new Instances(new FileReader("/home/slang/datasets/mnist_784_train_minimal.arff"));
        data.setClassIndex(data.numAttributes() - 1);

        OutputLayer outputLayer = new OutputLayer();
        outputLayer.setActivationFn(Activation.SOFTMAX.getActivationFunction());
        outputLayer.setWeightInit(WeightInit.XAVIER);
        outputLayer.setLayerName("Output-layer");
        outputLayer.setUpdater(Updater.ADAM);
        outputLayer.setLossFn(new LossMCXENT());

        NeuralNetConfiguration nnc = new NeuralNetConfiguration();
        nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);



        clf.setNeuralNetConfiguration(nnc);
        clf.setLayers(new Layer[]{outputLayer});

        TestUtil.holdout(clf, data);

    }


    


//    @Test
    public void testLeNetZooModel() throws IOException {
        LeNet model = new LeNet(10, 1, 1);
        model.setInputShape(new int[][]{new int[]{1, 28, 28}});
        org.deeplearning4j.nn.conf.MultiLayerConfiguration conf = model.conf();
        MultiLayerNetwork mln = new MultiLayerNetwork(conf);

        MnistDataSetIterator train = new MnistDataSetIterator(4, 1000, false, true, true, 1);
        MnistDataSetIterator test = new MnistDataSetIterator(4, 100, false, false, true, 1);

        while (train.hasNext()){
            DataSet next = train.next();
            System.out.println("next.numExamples() = " + next.numExamples());
            mln.fit(next);
            org.deeplearning4j.eval.Evaluation evaluate = mln.evaluate(test);
            System.out.println(evaluate.accuracy());
        }
    }

    @Test
    public void testLeNetWrapper() throws Exception {
        // CLF
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);

        // Data
        Instances data = DatasetLoader.loadMiniMnistMeta();
        data.setClassIndex(data.numAttributes() - 1);
        ImageDataSetIterator iterator = DatasetLoader.loadMiniMnistImageIterator();
        iterator.setTrainBatchSize(16);
        clf.setDataSetIterator(iterator);
        clf.setZooModel(new weka.dl4j.zoo.LeNet());
        clf.setNumEpochs(NUM_EPOCHS);
        clf.setIterationListener(new BatchListener());

        TestUtil.holdout(clf, data);
    }


    /**
     * Test minimal mnist conv net.
     *
     * @throws Exception IO error.
     */
    @Test
    public void testMinimalMnistConvNetArff() throws Exception {
        // CLF
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);
        clf.setNumEpochs(NUM_EPOCHS);
        clf.setDataSetIterator(new DefaultInstancesIterator());

        // Data
        Instances data = new Instances(new FileReader("/home/slang/datasets/mnist_784_train.arff"));
        data.setClassIndex(data.numAttributes() - 1);

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

        TestUtil.holdout(clf, data);

        final double[][] res = clf.distributionsForInstances(data);
        Assert.assertEquals(DatasetLoader.NUM_INSTANCES_MNIST, res.length);
        Assert.assertEquals(DatasetLoader.NUM_CLASSES_MNIST, res[0].length);
    }
}
