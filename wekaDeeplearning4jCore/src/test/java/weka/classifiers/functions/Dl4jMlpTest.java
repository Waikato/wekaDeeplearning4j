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
public class Dl4jMlpImageTest {

    /**
     * Logger instance
     */
    private static final Logger logger = LoggerFactory.getLogger(Dl4jMlpImageTest.class);


    /**
     * Default number of epochs
     */
    private static final int DEFAULT_NUM_EPOCHS = 10;

    /**
     * Seed
     */
    private static final int SEED = 42;

    /**
     * Default batch size
     */
    private static final int DEFAULT_BATCHSIZE = 32;


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
        clf.setSeed(SEED);
        clf.setNumEpochs(DEFAULT_NUM_EPOCHS);
        clf.setDebug(false);

        // Init data
        dataMnist = DatasetLoader.loadMiniMnistMeta();
        idiMnist = DatasetLoader.loadMiniMnistImageIterator();
        idiMnist.setTrainBatchSize(DEFAULT_BATCHSIZE);
        dataIris = DatasetLoader.loadIris();
        startTime = System.currentTimeMillis();
//        TestUtil.enableUIServer(clf);
    }


    @After
    public void after() throws IOException {

//        logger.info("Press anything to close");
//        Scanner sc = new Scanner(System.in);
//        sc.next();
        double time = (System.currentTimeMillis() - startTime)/1000.0;
        logger.info("Testmethod: " + name.getMethodName());
        logger.info("Time: " + time + "s");
    }

    /**
     * Test single layer iris.
     *
     * @throws Exception the exception
     */
    @Test
    public void testIris() throws Exception {
        // Define layers
        DenseLayer denseLayer = new DenseLayer();
        denseLayer.setNOut(32);
        denseLayer.setActivationFn(Activation.RELU.getActivationFunction());
        denseLayer.setWeightInit(WeightInit.XAVIER);
        denseLayer.setLayerName("Dense-layer");

        DenseLayer denseLayer2 = new DenseLayer();
        denseLayer2.setNOut(32);
        denseLayer2.setActivationFn(Activation.RELU.getActivationFunction());
        denseLayer2.setWeightInit(WeightInit.XAVIER);
        denseLayer2.setLayerName("Dense-layer");

        OutputLayer outputLayer = new OutputLayer();
        outputLayer.setActivationFn(Activation.SOFTMAX.getActivationFunction());
        outputLayer.setWeightInit(WeightInit.XAVIER);
        outputLayer.setUpdater(Updater.ADAM);
        outputLayer.setLearningRate(0.01);
        outputLayer.setBiasLearningRate(0.01);
        outputLayer.setLossFn(new LossMCXENT());

        // Configure neural network
        NeuralNetConfiguration nnc = new NeuralNetConfiguration();
        nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        nnc.setUseRegularization(true);

        clf.setNeuralNetConfiguration(nnc);
        clf.setLayers(new Layer[]{denseLayer, denseLayer2, outputLayer});

        TestUtil.holdout(clf, dataIris);
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
        ImageInstanceIterator imgIter = DatasetLoader.loadMiniMnistImageIterator();

        final int seed = 1;
        for (int batchSize : new int[]{1, 2, 5, 10}) {
            final int actual = countIterations(data, imgIter, seed, batchSize);
            final int expected = data.numInstances() / batchSize;
            Assert.assertEquals(expected, actual);
        }
    }

    /**
     * Counts the number of iterations an {@see ImageInstanceIterator}
     *
     * @param data      Instances to iterate
     * @param imgIter   ImageInstanceIterator to be tested
     * @param seed      Seed
     * @param batchsize Size of the batch which is returned
     *                  in {@see DataSetIterator#next}
     * @return Number of iterations
     * @throws Exception
     */
    private int countIterations(Instances data, ImageInstanceIterator imgIter,
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
        clf.setDataSetIterator(idiMnist);

        int[] threeByThree = {3, 3};
        int[] twoByTwo = {2, 2};
        int[] oneByOne = {1, 1};
        List<Layer> layers = new ArrayList<>();

        ConvolutionLayer convLayer1 = new ConvolutionLayer();
        convLayer1.setKernelSize(threeByThree);
        convLayer1.setStride(oneByOne);
        convLayer1.setNOut(32);
        convLayer1.setLayerName("Conv-layer 1");
        layers.add(convLayer1);

        BatchNormalization bn1 = new BatchNormalization();
        bn1.setActivationFunction(Activation.RELU.getActivationFunction());
        layers.add(bn1);

        ConvolutionLayer convLayer2 = new ConvolutionLayer();
        convLayer2.setKernelSize(threeByThree);
        convLayer2.setStride(oneByOne);
        convLayer2.setActivationFn(Activation.RELU.getActivationFunction());
        convLayer2.setNOut(32);
        layers.add(convLayer2);

        BatchNormalization bn2 = new BatchNormalization();
        bn2.setActivationFunction(Activation.RELU.getActivationFunction());
        layers.add(bn2);


        SubsamplingLayer poolLayer1 = new SubsamplingLayer();
        poolLayer1.setPoolingType(PoolingType.MAX);
        poolLayer1.setKernelSize(twoByTwo);
        poolLayer1.setLayerName("Pool1");
        layers.add(poolLayer1);


        ConvolutionLayer convLayer3 = new ConvolutionLayer();
        convLayer3.setNOut(64);
        convLayer3.setKernelSize(threeByThree);
        layers.add(convLayer3);

        BatchNormalization bn3 = new BatchNormalization();
        bn3.setActivationFunction(Activation.RELU.getActivationFunction());
        layers.add(bn3);

        ConvolutionLayer convLayer4 = new ConvolutionLayer();
        convLayer4.setNOut(64);
        convLayer4.setKernelSize(threeByThree);
        layers.add(convLayer4);

        BatchNormalization bn4 = new BatchNormalization();
        bn4.setActivationFunction(Activation.RELU.getActivationFunction());
        layers.add(bn4);

        SubsamplingLayer poolLayer2 = new SubsamplingLayer();
        poolLayer2.setPoolingType(PoolingType.MAX);
        poolLayer2.setKernelSize(twoByTwo);
        layers.add(poolLayer2);

        DenseLayer denseLayer1 = new DenseLayer();
        denseLayer1.setNOut(512);
        layers.add(denseLayer1);

        BatchNormalization bn5 = new BatchNormalization();
        bn5.setActivationFunction(Activation.RELU.getActivationFunction());
        bn5.setDropOut(0.2);
        layers.add(bn5);

        OutputLayer outputLayer = new OutputLayer();
        outputLayer.setActivationFn(Activation.SOFTMAX.getActivationFunction());
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
        clf.setDataSetIterator(idiMnist);

        DenseLayer denseLayer = new DenseLayer();
        denseLayer.setNOut(256);
        denseLayer.setLayerName("Dense-layer");
        denseLayer.setActivationFn(Activation.RELU.getActivationFunction());
        denseLayer.setWeightInit(WeightInit.XAVIER);

        DenseLayer denseLayer2 = new DenseLayer();
        denseLayer2.setNOut(128);
        denseLayer2.setLayerName("Dense-layer");
        denseLayer2.setActivationFn(Activation.RELU.getActivationFunction());
        denseLayer2.setWeightInit(WeightInit.XAVIER);

        OutputLayer outputLayer = new OutputLayer();
        outputLayer.setActivationFn(Activation.SOFTMAX.getActivationFunction());
        outputLayer.setLossFn(new LossMCXENT());
        outputLayer.setWeightInit(WeightInit.XAVIER);
        outputLayer.setLayerName("Output-layer");

        NeuralNetConfiguration nnc = new NeuralNetConfiguration();
        nnc.setOptimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        nnc.setPretrain(false);
        nnc.setSeed(SEED);

        clf.setNumEpochs(DEFAULT_NUM_EPOCHS);
        clf.setNeuralNetConfiguration(nnc);
        clf.setLayers(new Layer[]{denseLayer, denseLayer2, outputLayer});
        clf.addTrainingListener(new EpochListener());
        TestUtil.holdout(clf, dataMnist);
    }

    @Test
    public void testDl4j() throws Exception {
        //number of rows and columns in the input pictures
        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10; // number of output classes
        int batchSize = 64; // batch size for each epoch
        int rngSeed = 123; // random number seed for reproducibility
        int numEpochs = 15; // number of epochs to perform
        double rate = 0.0015; // learning rate
        int nChannels = 1; // Number of input channels

        //Get the DataSetIterators:
        ImageInstanceIterator iii = DatasetLoader.loadMiniMnistImageIterator();
        iii.setTrainBatchSize(DEFAULT_BATCHSIZE);
        Instances data = DatasetLoader.loadMiniMnistMeta();
        Instances[] split = TestUtil.splitTrainTest(data);
        Instances train = split[0];
        Instances test = split[1];
        DataSetIterator trainIt = iii.getIterator(train, SEED, DEFAULT_BATCHSIZE);
        DataSetIterator testIt = iii.getIterator(test, SEED, DEFAULT_BATCHSIZE);

        logger.info("Build model....");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .iterations(1)
                .learningRate(.01)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .list()
                .layer(0, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(256).build())
                .layer(1, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(128).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(28, 28, 1))
                .backprop(true).pretrain(false).build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new EvaluativeListener(iii.getIterator(train,SEED, DEFAULT_BATCHSIZE),1, InvocationType.EPOCH_END));  //print the score with every iteration

        logger.info("Train model....");
        for (int i = 0; i < DEFAULT_NUM_EPOCHS; i++) {
            logger.info("Epoch " + i);
            model.fit(trainIt);
        }


        logger.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
        while (testIt.hasNext()) {
            DataSet next = testIt.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }
        logger.info(eval.stats());
        logger.info("****************Example finished********************");
    }
}
