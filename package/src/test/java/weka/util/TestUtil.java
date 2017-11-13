package weka.util;

//import org.deeplearning4j.ui.api.UIServer;
//import org.deeplearning4j.ui.stats.StatsListener;
//import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.junit.Assert;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

        import java.util.Random;

/**
 * Utility class for evaluating classifier in JUnit tests
 *
 * @author Steven Lang
 */

public class TestUtil {

    /**
     * Logger instance
     */
    private static final Logger logger = LoggerFactory.getLogger(TestUtil.class);



    /**
     * Default number of epochs
     */
    public static final int DEFAULT_NUM_EPOCHS = 1;

    /**
     * Seed
     */
    public static final int SEED = 42;

    /**
     * Default batch size
     */
    public static final int DEFAULT_BATCHSIZE = 32;

    /**
     * Perform simple holdout (2/3, 1/3 split)
     *
     * @param clf  Classifier
     * @param data Full datase
     * @throws Exception
     */
    public static void holdout(Classifier clf, Instances data) throws Exception {
        Instances[] split = splitTrainTest(data);

        Instances train = split[0];
        Instances test = split[1];

        clf.buildClassifier(train);
        Evaluation trainEval = new Evaluation(train);
        logger.info("Train evaluation:");
        trainEval.evaluateModel(clf, train);
        logger.info(trainEval.toSummaryString());
        if (!data.classAttribute().isNumeric()) {
            logger.info(trainEval.toMatrixString());
        }


        logger.info("Test evaluation:");
        Evaluation testEval = new Evaluation(train);
        testEval.evaluateModel(clf, test);
        logger.info(testEval.toSummaryString());
        if (!data.classAttribute().isNumeric()) {
            logger.info(testEval.toMatrixString());
        }
//        if (testEval.pctCorrect() < testEval.pctIncorrect() * 2) {
//            Assert.fail("Too many incorrect predictions. " + "Correct% = " + testEval.pctCorrect() + "" +
//                    ", Incorrect% = " + testEval.pctIncorrect());
//
//        }
    }

    /**
     * Perform crossvalidation
     *
     * @param clf  Classifier
     * @param data Full dataset
     * @throws Exception
     */
    public static void crossValidate(Classifier clf, Instances data) throws Exception {
        Evaluation ev = new Evaluation(data);
        ev.crossValidateModel(clf, data, 10, new Random(42));
        logger.info(ev.toSummaryString());
    }

    /**
     * Split the dataset into p% traind an (100-p)% test set
     *
     * @param data Input data
     * @param p    train percentage
     * @return Array of instances: (0) Train, (1) Test
     * @throws Exception Filterapplication went wrong
     */
    public static Instances[] splitTrainTest(Instances data, double p) throws Exception {

        Randomize rand = new Randomize();
        rand.setInputFormat(data);
        rand.setRandomSeed(42);
        data = Filter.useFilter(data, rand);

        RemovePercentage rp = new RemovePercentage();
        rp.setInputFormat(data);
        rp.setPercentage(p);
        Instances train = Filter.useFilter(data, rp);


        rp = new RemovePercentage();
        rp.setInputFormat(data);
        rp.setPercentage(p);
        rp.setInvertSelection(true);
        Instances test = Filter.useFilter(data, rp);

        return new Instances[]{train, test};
    }

    /**
     * Split the dataset into 67% traind an 33% test set
     *
     * @param data Input data
     * @return Array of instances: (0) Train, (1) Test
     * @throws Exception Filterapplication went wrong
     */
    public static Instances[] splitTrainTest(Instances data) throws Exception {
        return splitTrainTest(data, 33);
    }

    /**
     * Convert the classifier to commandline arguments
     * @param clf Classifier
     * @return CLF-String formatted as commandline argument
     */
    public static String toCmdLineArgs(Dl4jMlpClassifier clf){
        String[] opts = clf.getOptions();
        String res = "";
        for (int i = 0; i < opts.length; i+=2) {
            if (opts[i+1].equals("NaN"))continue;
            res += opts[i] + " \"" + opts[i+1].replace("\"","\\\"") + "\" \\\n" ;
        }
        return res;
    }

//
//    /**
//     * Enables the UIServer at http://localhost:9000/train
//     *
//     * @param clf Dl4jMlpClassifier instance
//     */
//    public static void enableUIServer(Dl4jMlpClassifier clf) {
//        //Initialize the user interface backend
//        UIServer uiServer = UIServer.getInstance();
//
//        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
//        File f = new File("/tmp/out.bin");
//        f.delete();
//        StatsStorage statsStorage = new FileStatsStorage(f);
//
//        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
//        uiServer.attach(statsStorage);
//
//        //Then add the StatsListener to collect this information from the network, as it trains
//        clf.addTrainingListener(new StatsListener(statsStorage));
//    }
}
