package weka.classifiers.functions;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestName;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.core.*;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.activations.ActivationReLU;
import weka.dl4j.activations.ActivationSoftmax;
import weka.dl4j.earlystopping.EarlyStopping;
import weka.dl4j.iterators.instance.ConvolutionInstanceIterator;
import weka.dl4j.iterators.instance.DefaultInstanceIterator;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.layers.*;
import weka.dl4j.listener.EpochListener;
import weka.dl4j.lossfunctions.LossMCXENT;
import weka.dl4j.zoo.LeNet;
import weka.util.DatasetLoader;
import weka.util.TestUtil;

import java.io.*;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


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
        clf.setNumEpochs(TestUtil.DEFAULT_NUM_EPOCHS);
        clf.setEarlyStopping(new EarlyStopping(5, 15));

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

    /**
     * Test Layer setup DENSE to CONV which is currently not supported
     *
     * @throws Exception Could not build classifier.
     */
    @Test(expected = RuntimeException.class)
    public void testIllegalIrisConv() throws Exception {
        final ConvolutionInstanceIterator it = new ConvolutionInstanceIterator();
        it.setHeight(1);
        it.setWidth(4);
        clf.setInstanceIterator(it);


        ConvolutionLayer cl = new ConvolutionLayer();
        cl.setNOut(3);
        cl.setKernelSize(new int[]{1, 1});
        cl.setStride(new int[]{1, 1});


        DenseLayer dl = new DenseLayer();
        dl.setNOut(10);

        OutputLayer ol = new OutputLayer();
        clf.setLayers(new Layer[]{dl, cl, ol});
        clf.buildClassifier(dataIris);
    }

    /**
     * Test convolution while setting {@link DefaultInstanceIterator} which is
     * forbidden.
     *
     * @throws Exception Could not build classifier.
     */
    @Test(expected = RuntimeException.class)
    public void testIllegalIrisConvDefaultInstanceIterator() throws Exception {
        // DefaultInstanceIterator should not be allowed in the combination with
        // convolutional layers.
        clf.setInstanceIterator(new DefaultInstanceIterator());


        DenseLayer dl = new DenseLayer();
        dl.setNOut(10);

        ConvolutionLayer cl = new ConvolutionLayer();
        cl.setNOut(3);
        cl.setKernelSize(new int[]{1, 1});
        cl.setStride(new int[]{1, 1});

        OutputLayer ol = new OutputLayer();
        clf.setLayers(new Layer[]{cl, dl, ol});
        clf.buildClassifier(dataIris);
    }

    /**
     * Test iris convolution.
     *
     * @throws Exception Could not build classifier.
     */
    @Test
    public void testIrisConv() throws Exception {
        final ConvolutionInstanceIterator it = new ConvolutionInstanceIterator();
        it.setHeight(1);
        it.setWidth(4);
        clf.setInstanceIterator(it);

        DenseLayer dl = new DenseLayer();
        dl.setNOut(10);

        ConvolutionLayer cl = new ConvolutionLayer();
        cl.setNOut(3);
        cl.setKernelSize(new int[]{1, 1});
        cl.setStride(new int[]{1, 1});

        OutputLayer ol = new OutputLayer();
        clf.setLayers(new Layer[]{cl, dl, ol});
        clf.buildClassifier(dataIris);
    }

    /**
     * Test to serialization of the classifier. This is important for the GUI
     * usage.
     *
     * @throws Exception Could not build classifier.
     */
    @Test
    public void testSerialization() throws Exception {
        clf.setInstanceIterator(idiMnist);

        DenseLayer denseLayer = new DenseLayer();
        denseLayer.setNOut(8);
        denseLayer.setLayerName("Dense-layer");
        denseLayer.setActivationFn(new ActivationReLU());

        DenseLayer denseLayer2 = new DenseLayer();
        denseLayer2.setNOut(4);
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

        File out = Paths.get(System.getProperty("java.io.tmpdir"), "out.object").toFile();
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(out));
        oos.writeObject(clf);

        ObjectInputStream ois = new ObjectInputStream(new FileInputStream(out));
        Dl4jMlpClassifier clf2 = (Dl4jMlpClassifier) ois.readObject();

        clf2.setNumEpochs(1);
        clf2.initializeClassifier(dataMnist);
        clf2.buildClassifier(dataMnist);
    }

    /**
     * Test UnsupportedAttributeTypeException
     * @throws Exception
     */
    @Test(expected = UnsupportedAttributeTypeException.class)
    public void testWrongArffFormat() throws Exception {
        Attribute att1 = new Attribute("1", true);
        Attribute att2 = new Attribute("2", Arrays.asList("1","2"));
        ArrayList<Attribute> atts = new ArrayList<>();
        atts.add(att1);
        atts.add(att2);
        Instances inst = new Instances("", atts, 10);
        Instance ins = new DenseInstance(1);
        ins.setDataset(inst);
        inst.setClassIndex(0);
        ins.setValue(0, "1");
        inst.add(ins);
        clf.initializeClassifier(inst);
    }

    /**
     * Test no outputlayer
     * @throws MissingOutputLayerException
     */
    @Test(expected = MissingOutputLayerException.class)
    public void testLastLayerNoOutputLayer() throws Exception {
        clf.setLayers(new Layer[]{new DenseLayer()});
        clf.initializeClassifier(dataIris);
    }

    /**
     * Test async iterator
     */
    @Test
    public void testAsyncIterator() throws Exception {
        clf.setQueueSize(4);
        clf.buildClassifier(dataIris);
    }
    /**
     * Test zoo model with wrong iterator
     */
    @Test(expected = WrongIteratorException.class)
    public void testZooModelWithoutImageIterator() throws Exception {
        clf.setZooModel(new LeNet());
        clf.setInstanceIterator(new DefaultInstanceIterator());
        clf.buildClassifier(dataIris);
    }
}
