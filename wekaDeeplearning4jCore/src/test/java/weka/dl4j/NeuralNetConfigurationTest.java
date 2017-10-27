package weka.dl4j;

import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.*;
import org.junit.rules.TestName;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.Instances;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.layers.*;
import weka.util.DatasetLoader;

import java.io.IOException;


/**
 * JUnit tests for the NeuralNetConfiguration.
 * Tests setting parameters and different configurations.
 *
 * @author Steven Lang
 */
public class NeuralNetConfigurationTest {

    /**
     * Logger instance
     */
    private static final Logger logger = LoggerFactory.getLogger(NeuralNetConfigurationTest.class);


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
        startTime = System.currentTimeMillis();
        clf.setInstanceIterator(idiMnist);
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
     * Test
     * @throws Exception
     */
    @Test
    public void testNNCParameters() throws Exception {

        for (Updater u : Updater.values()) {
            // Define some architecture including all currently available layers
            ConvolutionLayer cl = new ConvolutionLayer();
            cl.setNOut(10);
            DenseLayer dl = new DenseLayer();
            dl.setNOut(10);
            BatchNormalization bn = new BatchNormalization();
            OutputLayer ol = new OutputLayer();
            Layer[] layers = {dl, bn, ol};
            clf.setLayers(layers);

            // Setup the configuration
            NeuralNetConfiguration nnc;

            // Skip custom
            if (u.equals(Updater.CUSTOM)) continue;

            // NADAM is not working as of dl4j 0.9.1
            if (u.equals(Updater.NADAM)) continue;

            double l1 = 0.001;
            double l2 = 0.002;
            double l1Bias = 0.003;
            double l2Bias = 0.004;

            int learningRate = 1;
            WeightInit weightInit = WeightInit.UNIFORM;

            nnc = new NeuralNetConfiguration();
            nnc.setUseRegularization(true);
            nnc.setUpdater(u);
            nnc.setL1(l1);
            nnc.setL2(l2);

            // Not working as of dl4j 0.9.1
//            nnc.setBiasL1(l1Bias);
//            nnc.setBiasL2(l2Bias);
            nnc.setLearningRate(learningRate);
            nnc.setWeightInit(weightInit);


            clf.setNeuralNetConfiguration(nnc);
            clf.initializeClassifier(dataMnist); // creates the model internally

            for (Layer l : layers) {
                Updater u2 = ((BaseLayer) l).getUpdater();
                double l11 = ((BaseLayer) l).getL1();
                double l21 = ((BaseLayer) l).getL2();
                double learningRate1 = ((BaseLayer) l).getLearningRate();
                WeightInit weightInit1 = ((BaseLayer) l).getWeightInit();

                Assert.assertEquals(u.name(), u2.name());
                Assert.assertEquals(l1, l11, 10e-6);
                Assert.assertEquals(l2, l21, 10e-6);
                Assert.assertEquals(learningRate, learningRate1, 10e-6);
                Assert.assertEquals(weightInit, weightInit1);
                // Not working as of dl4j 0.9.1
//                Assert.assertEquals(l1Bias, ((BaseLayer) l).getL1Bias(), 10e-6);
//                Assert.assertEquals(l2Bias, ((BaseLayer) l).getL2Bias(), 10e-6);
            }
        }
    }
}
