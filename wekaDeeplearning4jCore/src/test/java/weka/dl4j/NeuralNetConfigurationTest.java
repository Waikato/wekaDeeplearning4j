package weka.dl4j;

import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.*;
import org.junit.rules.TestName;
import org.nd4j.linalg.learning.config.IUpdater;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.Instances;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.layers.BatchNormalization;
import weka.dl4j.layers.ConvolutionLayer;
import weka.dl4j.layers.DenseLayer;
import weka.dl4j.layers.OutputLayer;
import weka.dl4j.updater.*;
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
     *
     * @throws Exception
     */
    @Test
    public void testNNCParameters() throws Exception {
        weka.dl4j.updater.Updater[] updater = new weka.dl4j.updater.Updater[]{
                new AdaDelta(),
                new AdaGrad(),
                new Adam(),
                new AdaMax(),
                new Nadam(),
                new Nesterovs(),
                new NoOp(),
                new RmsProp(),
                new Sgd()
        };
        for (weka.dl4j.updater.Updater u : updater) {
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

            double l1 = 0.001;
            double l2 = 0.002;
            double l1Bias = 0.003;
            double l2Bias = 0.004;
            if (u instanceof Adam) {
                double lr = 5.0;
                ((Adam) u).setLearningRate(lr);
            }
            WeightInit weightInit = WeightInit.UNIFORM;

            nnc = new NeuralNetConfiguration();
            nnc.setUseRegularization(true);
            nnc.setUpdater(u);

            nnc.setL1(l1);
            nnc.setL2(l2);

            // Not working as of dl4j 0.9.1
//            nnc.setBiasL1(l1Bias);
//            nnc.setBiasL2(l2Bias);
            nnc.setWeightInit(weightInit);


            clf.setNeuralNetConfiguration(nnc);
            clf.initializeClassifier(dataMnist); // creates the model internally

            for (Layer l : layers) {
                IUpdater u2 = ((BaseLayer) l).getIUpdater();
                double l11 = ((BaseLayer) l).getL1();
                double l21 = ((BaseLayer) l).getL2();
                WeightInit weightInit1 = ((BaseLayer) l).getWeightInit();
                double learningRate = ((BaseLayer) l).getLearningRate();
                if (u instanceof Adam) {
                    Assert.assertEquals(5.0, learningRate, 10e-6);
                }

                Assert.assertEquals(u.getClass().getSimpleName(), u2.getClass().getSimpleName());
                Assert.assertEquals(l1, l11, 10e-6);
                Assert.assertEquals(l2, l21, 10e-6);
                Assert.assertEquals(weightInit, weightInit1);
            }
        }
    }
}
