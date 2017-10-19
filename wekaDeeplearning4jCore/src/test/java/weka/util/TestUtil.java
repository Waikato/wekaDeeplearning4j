package weka.util;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

public class TestUtil {

    /**
     * Perform simple holdout (2/3, 1/3 split)
     * @param clf Classifier
     * @param data Full dataset
     * @throws Exception
     */
    public static void holdout(Classifier clf, Instances data) throws Exception {
        Instances[] split = splitTrainTest(data);

        clf.buildClassifier(split[0]);
        Evaluation eval = new Evaluation(split[0]);
        System.out.println("Train evaluation:");
        eval.evaluateModel(clf, split[0]);
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());


        System.out.println("Test evaluation:");
        eval = new Evaluation(split[0]);
        eval.evaluateModel(clf, split[1]);
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());
    }

    /**
     * Split the dataset into p% traind an (100-p)% test set
     * @param data Input data
     * @param p train percentage
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
     * @param data Input data
     * @return Array of instances: (0) Train, (1) Test
     * @throws Exception Filterapplication went wrong
     */
    public static Instances[] splitTrainTest(Instances data) throws Exception {
        return splitTrainTest(data, 33);
    }
}
