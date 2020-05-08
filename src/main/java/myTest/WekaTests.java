package myTest;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.PoolingType;
import weka.dl4j.PretrainedType;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.updater.Adam;
import weka.dl4j.updater.Updater;
import weka.dl4j.zoo.*;
import weka.dl4j.zoo.Dl4jVGG;
import weka.dl4j.zoo.keras.DenseNet;
import weka.dl4j.zoo.keras.NASNet;
import weka.dl4j.zoo.keras.ResNet;
import weka.dl4j.zoo.keras.VGG;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Dl4jMlpFilter;

import java.io.File;
import java.io.FileReader;
import java.util.Random;

public class WekaTests {
    public WekaTests() {}

    public void filterExample(String[] args) throws Exception {
// Load all packages so that Dl4jMlpFilter class can be found using forName("weka.filters.unsupervised.attribute.Dl4jMlpFilter")
        weka.core.WekaPackageManager.loadPackages(true);

// Load the dataset
        weka.core.Instances instances = new weka.core.Instances(new FileReader("E:\\Rhys\\Documents\\Github\\wekaDeeplearning4j\\datasets\\nominal\\mnist.meta.minimal.arff"));
        instances.setClassIndex(1);
        String[] filterOptions = weka.core.Utils.splitOptions("-iterator \".ImageInstanceIterator -imagesLocation E:\\\\Rhys\\\\Documents\\\\Github\\\\wekaDeeplearning4j\\\\datasets\\\\nominal\\\\mnist-minimal -bs 12\" -poolingType AVG -layer-extract \"weka.dl4j.layers.DenseLayer -name res4a_branch2b\" -layer-extract \"weka.dl4j.layers.DenseLayer -name flatten_1\" -isZoo true -zooModel \".Dl4JResNet50\"");
        weka.filters.Filter myFilter = (weka.filters.Filter) weka.core.Utils.forName(weka.filters.Filter.class, "weka.filters.unsupervised.attribute.Dl4jMlpFilter", filterOptions);
//        myFilter.setOptions();

// Run the filter, using the model as a feature extractor
        myFilter.setInputFormat(instances);
        weka.core.Instances transformedInstances = weka.filters.Filter.useFilter(instances, myFilter);

// You could save the instances at this point to an arff file to be used with other classifiers via:
// https://waikato.github.io/weka-wiki/formats_and_processing/save_instances_to_arff/

// CV our Random Forest classifier on the extracted features
        weka.classifiers.evaluation.Evaluation evaluation = new weka.classifiers.evaluation.Evaluation(transformedInstances);
        int numFolds = 10;
        evaluation.crossValidateModel(new weka.classifiers.trees.RandomForest(), transformedInstances, numFolds, new Random(1));
        System.out.println(evaluation.toSummaryString());
        System.out.println(evaluation.toMatrixString());
    }

    public void filterTest(String[] args) throws Exception {
        Dl4jMlpFilter myFilter = new Dl4jMlpFilter();
        ImageInstanceIterator imgIter = new ImageInstanceIterator();
        imgIter.setImagesLocation(new File("datasets/nominal/mnist-minimal"));
        imgIter.setTrainBatchSize(16);
        myFilter.setImageInstanceIterator(imgIter);
        Dl4jSimpleCNN zooModel = new Dl4jSimpleCNN();
//        zooModel.setVariation(NASNet.VARIATION.MOBILE);
//        myFilter.addTransformationLayerName("reduction_conv_1_stem_1");
        myFilter.setZooModelType(zooModel);
        Filter.runFilter(myFilter, args);
    }

    public void train(String[] args) throws Exception {
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);
        clf.setNumEpochs(0);

        // Load the arff file
        Instances data = new Instances(new FileReader("datasets\\nominal\\mnist.meta.minimal.arff"));
        data.setClassIndex(data.numAttributes() - 1);

        ImageInstanceIterator imgIter = new ImageInstanceIterator();
        imgIter.setImagesLocation(new File("datasets\\nominal\\mnist-minimal"));
        imgIter.setTrainBatchSize(16);
        clf.setInstanceIterator(imgIter);

        // Set up the network configuration
        NeuralNetConfiguration nnc = new NeuralNetConfiguration();
        Updater updater = new Adam();
        updater.setLearningRate(0.1);
        nnc.setUpdater(updater);
        clf.setNeuralNetConfiguration(nnc);

        // Set up the pretrained model
        Dl4jAlexNet zooModel = new Dl4jAlexNet();
//        zooModel.setVariation(Dl4jDarknet19.VARIATION.INPUT448);
        clf.setZooModel(zooModel);

        // Stratify and split the data
        Random rand = new Random(0);
        Instances randData = new Instances(data);
        randData.randomize(rand);
        randData.stratify(3);
        Instances train = randData.trainCV(3, 0);
        Instances test = randData.testCV(3, 0);

        // Build the classifier on the training data
        clf.buildClassifier(train);

        // Evaluate the model on test data
        Evaluation eval = new Evaluation(test);
        eval.evaluateModel(clf, test);

        // Output some summary statistics
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());
    }

    public static void main(String[] args) throws Exception {
        new WekaTests().filterTest(args);
    }
}
