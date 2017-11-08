package weka.classifiers.functions;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.junit.*;
import org.junit.rules.TestName;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.core.Instances;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.activations.ActivationReLU;
import weka.dl4j.activations.ActivationSoftmax;
import weka.dl4j.earlystopping.EarlyStopping;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.layers.*;
import weka.dl4j.listener.EpochListener;
import weka.dl4j.lossfunctions.LossMCXENT;
import weka.util.DatasetLoader;
import weka.util.TestUtil;

import java.io.IOException;
import java.util.*;


/**
 * JUnit tests for the Dl4jMlpClassifier.
 * Tests nominal classes with iris, numerical classes with diabetes and image
 * classification with minimal mnist.
 *
 * @author Steven Lang
 */
public class Dl4jMlpTest {

    /**
     * Logger instance
     */
    private static final Logger logger = LoggerFactory.getLogger(Dl4jMlpTest.class);



    /**
     * Classifier
     */
    private Dl4jMlpClassifier clf;

    /**
     * Dataset mnist
     */
    private Instances dataMnist;

    /**
     * Mnist image loader
     */
    private ImageInstanceIterator idiMnist;

    /**
     * Dataset iris
     */
    private Instances dataIris;

    /**
     * Current name
     */
    @Rule
    public TestName name = new TestName();

    /**
     * Start time for time measurement
     */
    private long startTime;

    @Before
    public void before() throws Exception {
        // Init mlp clf
        clf = new Dl4jMlpClassifier();
        clf.setSeed(TestUtil.SEED);
        clf.setNumEpochs(TestUtil.DEFAULT_NUM_EPOCHS*2);
        clf.setEarlyStoppingConfiguration(new EarlyStopping(5, 15));

        clf.setDebug(false);

        // Init data
        dataMnist = DatasetLoader.loadMiniMnistMeta();
        idiMnist = DatasetLoader.loadMiniMnistImageIterator();
        idiMnist.setTrainBatchSize(TestUtil.DEFAULT_BATCHSIZE);
        dataIris = DatasetLoader.loadIris();
        startTime = System.currentTimeMillis();
//        TestUtil.enableUIServer(clf);
    }


    @After
    public void after() throws IOException {

//        logger.info("Press anything to close");
//        Scanner sc = new Scanner(System.in);
//        sc.next();
        double time = (System.currentTimeMillis() - startTime) / 1000.0;
        logger.info("Testmethod: " + name.getMethodName());
        logger.info("Time: " + time + "s");
    }


    /**
     * Test minimal mnist conv net.
     *
     * @throws Exception IO error.
     */
    @Test
    public void testMinimalMnistConvNet() throws Exception {
        clf.setInstanceIterator(idiMnist);

        int[] threeByThree = {3, 3};
        int[] twoByTwo = {2, 2};
        int[] oneByOne = {1, 1};
        List<Layer> layers = new ArrayList<>();

        ConvolutionLayer convLayer1 = new ConvolutionLayer();
        convLayer1.setKernelSize(threeByThree);
        convLayer1.setStride(oneByOne);
        convLayer1.setNOut(8);
        convLayer1.setLayerName("Conv-layer 1");
        layers.add(convLayer1);

        SubsamplingLayer poolLayer1 = new SubsamplingLayer();
        poolLayer1.setPoolingType(PoolingType.MAX);
        poolLayer1.setKernelSize(twoByTwo);
        poolLayer1.setLayerName("Pool1");
        layers.add(poolLayer1);


        ConvolutionLayer convLayer3 = new ConvolutionLayer();
        convLayer3.setNOut(8);
        convLayer3.setKernelSize(threeByThree);
        layers.add(convLayer3);

        BatchNormalization bn4 = new BatchNormalization();
        bn4.setActivationFunction(new ActivationReLU());
        layers.add(bn4);

        SubsamplingLayer poolLayer2 = new SubsamplingLayer();
        poolLayer2.setPoolingType(PoolingType.MAX);
        poolLayer2.setKernelSize(twoByTwo);
        layers.add(poolLayer2);

        OutputLayer outputLayer = new OutputLayer();
        outputLayer.setActivationFn(new ActivationSoftmax());
        outputLayer.setLossFn(new LossMCXENT());
        layers.add(outputLayer);

        NeuralNetConfiguration nnc = new NeuralNetConfiguration();
        nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        nnc.setUseRegularization(true);

        clf.setNeuralNetConfiguration(nnc);
        Layer[] ls = new Layer[layers.size()];
        layers.toArray(ls);
        clf.setLayers(ls);

        TestUtil.holdout(clf, dataMnist);
    }


    /**
     * Test minimal mnist dense net.
     *
     * @throws Exception IO error.
     */
    @Test
    public void testMinimalMnistDense() throws Exception {
        clf.setInstanceIterator(idiMnist);

        DenseLayer denseLayer = new DenseLayer();
        denseLayer.setNOut(128);
        denseLayer.setLayerName("Dense-layer");
        denseLayer.setActivationFn(new ActivationReLU());

        DenseLayer denseLayer2 = new DenseLayer();
        denseLayer2.setNOut(32);
        denseLayer2.setLayerName("Dense-layer");
        denseLayer2.setActivationFn(new ActivationReLU());

        OutputLayer outputLayer = new OutputLayer();
        outputLayer.setActivationFn(new ActivationSoftmax());
        outputLayer.setLossFn(new LossMCXENT());
        outputLayer.setLayerName("Output-layer");

        NeuralNetConfiguration nnc = new NeuralNetConfiguration();
        nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        nnc.setPretrain(false);
        nnc.setSeed(TestUtil.SEED);

        clf.setNeuralNetConfiguration(nnc);
        clf.setLayers(new Layer[]{denseLayer, denseLayer2, outputLayer});
        clf.setIterationListener(new EpochListener());
        TestUtil.holdout(clf, dataMnist);
    }
}
