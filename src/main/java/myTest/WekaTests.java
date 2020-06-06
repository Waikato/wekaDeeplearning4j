package myTest;
//        KerasLayer.registerLambdaLayer("broadcast_w112_d32_1", new CustomBroadcast(112));
//        KerasLayer.registerLambdaLayer("broadcast_w56_d96_1", new CustomBroadcast(56));
//        KerasLayer.registerLambdaLayer("broadcast_w56_d144_1", new CustomBroadcast(56));
//        KerasLayer.registerLambdaLayer("broadcast_w28_d144_1", new CustomBroadcast(28));
//        KerasLayer.registerLambdaLayer("broadcast_w28_d240_1", new CustomBroadcast(28));
//        KerasLayer.registerLambdaLayer("broadcast_w14_d240_1", new CustomBroadcast(14));
//        KerasLayer.registerLambdaLayer("broadcast_w14_d480_1", new CustomBroadcast(14));
//        KerasLayer.registerLambdaLayer("broadcast_w14_d480_2", new CustomBroadcast(14));
//        KerasLayer.registerLambdaLayer("broadcast_w14_d480_3", new CustomBroadcast(14));
//        KerasLayer.registerLambdaLayer("broadcast_w14_d672_1", new CustomBroadcast(14));
//        KerasLayer.registerLambdaLayer("broadcast_w14_d672_2", new CustomBroadcast(14));
//        KerasLayer.registerLambdaLayer("broadcast_w7_d672_1", new CustomBroadcast(7));
//        KerasLayer.registerLambdaLayer("broadcast_w7_d1152_1", new CustomBroadcast(7));
//        KerasLayer.registerLambdaLayer("broadcast_w7_d1152_2", new CustomBroadcast(7));
//        KerasLayer.registerLambdaLayer("broadcast_w7_d1152_3", new CustomBroadcast(7));
//        KerasLayer.registerLambdaLayer("broadcast_w7_d1152_4", new CustomBroadcast(7));

import com.sun.jna.platform.win32.OaIdl;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.zoo.model.Darknet19;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.AllJavadoc;
import weka.core.Instances;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.PretrainedType;
import weka.dl4j.iterators.instance.ConvolutionInstanceIterator;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.layers.lambda.CustomBroadcast;
import weka.dl4j.updater.Adam;
import weka.dl4j.zoo.*;
import weka.dl4j.zoo.Dl4jVGG;
import weka.dl4j.zoo.keras.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Dl4jMlpFilter;

import java.io.*;
import java.util.*;

public class WekaTests {

    public void filterExampleReflectionMNIST(String[] args) throws Exception {
        // Load all packages so that Dl4jMlpFilter class can be found using forName("weka.filters.unsupervised.attribute.Dl4jMlpFilter")
        weka.core.WekaPackageManager.loadPackages(true);

        // Load the dataset
        weka.core.Instances instances = new weka.core.Instances(new FileReader("datasets/nominal/mnist.meta.minimal.arff"));
        instances.setClassIndex(1);
        String[] filterOptions = weka.core.Utils.splitOptions("-iterator \".ImageInstanceIterator -imagesLocation datasets/nominal/mnist-minimal -bs 12\" -poolingType AVG -layer-extract \"weka.dl4j.layers.DenseLayer -name flatten_1\" -layer-extract \"weka.dl4j.layers.DenseLayer -name res4a_branch2b\" -zooModel \".Dl4JResNet50\"");
        weka.filters.Filter myFilter = (weka.filters.Filter) weka.core.Utils.forName(weka.filters.Filter.class, "weka.filters.unsupervised.attribute.Dl4jMlpFilter", filterOptions);

        // Run the filter, using the model as a feature extractor
        myFilter.setInputFormat(instances);
        weka.core.Instances transformedInstances = weka.filters.Filter.useFilter(instances, myFilter);

        // You could save the instances at this point to an arff file for rapid experimentation with other classifiers via:
        // https://waikato.github.io/weka-wiki/formats_and_processing/save_instances_to_arff/

        // CV our Random Forest classifier on the extracted features
        weka.classifiers.evaluation.Evaluation evaluation = new weka.classifiers.evaluation.Evaluation(transformedInstances);
        int numFolds = 10;
        evaluation.crossValidateModel(new weka.classifiers.functions.SMO(), transformedInstances, numFolds, new Random(1));
        System.out.println(evaluation.toSummaryString());
        System.out.println(evaluation.toMatrixString());
    }

    public void filterExampleReflection(String[] args) throws Exception {
        // Load all packages so that Dl4jMlpFilter class can be found using forName("weka.filters.unsupervised.attribute.Dl4jMlpFilter")
        weka.core.WekaPackageManager.loadPackages(true);

        // Load the dataset
        weka.core.Instances data = new weka.core.Instances(new FileReader("/home/rhys/Documents/git/kaggle-competitions/plant-seedlings/data/train/output.arff"));
        data.setClassIndex(data.numAttributes() - 1);
        String[] classifierOptions = weka.core.Utils.splitOptions("-iterator \"weka.dl4j.iterators.instance.ImageInstanceIterator -imagesLocation /home/rhys/Documents/git/kaggle-competitions/plant-seedlings/data/train -bs 16\" -zooModel \"weka.dl4j.zoo.KerasResNet -variation RESNET152V2\" -config \"weka.dl4j.NeuralNetConfiguration -updater \\\"weka.dl4j.updater.Adam -lr 0.1\\\"\" -numEpochs 20");
        weka.classifiers.AbstractClassifier myClassifier = (AbstractClassifier) weka.core.Utils.forName(weka.classifiers.AbstractClassifier.class, "weka.classifiers.functions.Dl4jMlpClassifier", classifierOptions);

        // Stratify and split the data
        Random rand = new Random(0);
        Instances randData = new Instances(data);
        randData.randomize(rand);
        randData.stratify(5);
        Instances train = randData.trainCV(5, 0);
        Instances test = randData.testCV(5, 0);

        // Build the classifier on the training data
        myClassifier.buildClassifier(train);

        // Evaluate the model on test data
        Evaluation eval = new Evaluation(test);
        eval.evaluateModel(myClassifier, test);

        // Output some summary statistics
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());
    }

    public void filterExample(String[] args) throws Exception {
        // Load the dataset
        Instances instances = new Instances(new FileReader("datasets/nominal/mnist.meta.minimal.arff"));
        instances.setClassIndex(1);
        Dl4jMlpFilter myFilter = new Dl4jMlpFilter();

// Create our iterator, pointing it to the location of the images
        ConvolutionInstanceIterator imgIter = new ConvolutionInstanceIterator();
        imgIter.setHeight(28);
        imgIter.setWidth(28);
        imgIter.setNumChannels(1);
//        imgIter.setImagesLocation(new File("datasets/nominal/mnist-minimal"));
// Set batch size to thread count if using CPU
        imgIter.setTrainBatchSize(12);
        myFilter.setInstanceIterator(imgIter);

// Load our pretrained model
        Dl4JResNet50 zooModel = new Dl4JResNet50();
        myFilter.setZooModelType(zooModel);

// Run the filter, using the model as a feature extractor
        myFilter.setInputFormat(instances);
        Instances transformedInstances = Filter.useFilter(instances, myFilter);

// You could save the instances at this point to an arff file to be used with other classifiers via:
// https://waikato.github.io/weka-wiki/formats_and_processing/save_instances_to_arff/

// CV our Random Forest classifier on the extracted features
        Evaluation evaluation = new Evaluation(transformedInstances);
        int numFolds = 10;
        evaluation.crossValidateModel(new RandomForest(), transformedInstances, numFolds, new Random(1));
        System.out.println(evaluation.toSummaryString());
        System.out.println(evaluation.toMatrixString());
    }

    public void filterTest(String[] args) throws Exception {
            Dl4jMlpFilter myFilter = new Dl4jMlpFilter();

            ConvolutionInstanceIterator imgIter = new ConvolutionInstanceIterator();
            imgIter.setHeight(28);
            imgIter.setWidth(28);
            imgIter.setNumChannels(1);
            imgIter.setTrainBatchSize(8);

            myFilter.setInstanceIterator(imgIter);

            Dl4jLeNet thisModel = new Dl4jLeNet();
//            thisModel.setVariation(ResNet.VARIATION.RESNET101);
//            thisModel.setVariation(NASNet.VARIATION.LARGE);
//            thisModel.setVariation(NASNet.VARIATION.LARGE);
            myFilter.setZooModelType(thisModel);

            Filter.runFilter(myFilter, args);
    }

    public void train(String[] args) throws Exception {
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);
        clf.setNumEpochs(30);

        // Load the arff file
        Instances data = new Instances(new FileReader("datasets/nominal/mnist.meta.tiny.arff"));
//                Instances data = new Instances(new FileReader("/home/rhys/Documents/datasets/mnist_784/mnist_784_test_small.arff"));
        data.setClassIndex(data.numAttributes() - 1);

        ImageInstanceIterator imgIter = new ImageInstanceIterator();
        imgIter.setImagesLocation(new File("datasets/nominal/mnist-minimal"));
        imgIter.setTrainBatchSize(16);
//        imgIter.setWidth(28);
//        imgIter.setHeight(28);
//        imgIter.setNumChannels(1);
        clf.setInstanceIterator(imgIter);

        // Set up the pretrained model
        Dl4jSqueezeNet zooModel = new Dl4jSqueezeNet();
        zooModel.setVariation(Dl4jDarknet19.VARIATION.INPUT224);
//        zooModel.setVariation(NASNet.VARIATION.LARGE);
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
        new WekaTests().train(args);
    }
}
