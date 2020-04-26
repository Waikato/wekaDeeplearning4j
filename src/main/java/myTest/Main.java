package myTest;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.PoolingType;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.updater.Adam;
import weka.dl4j.updater.Updater;
import weka.dl4j.zoo.*;
import weka.dl4j.zoo.Dl4jVGG;
import weka.dl4j.zoo.keras.DenseNet;
import weka.dl4j.zoo.keras.NASNet;
import weka.dl4j.zoo.keras.ResNet;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Dl4jMlpFilter;

import java.io.File;
import java.io.FileReader;
import java.util.Random;

class WekaTests {
    public WekaTests() {}

    public void filterExample(String[] args) throws Exception {
        // Load the dataset
        Instances instances = new Instances(new FileReader("datasets/nominal/mnist.meta.minimal.arff"));
        instances.setClassIndex(1);
        Dl4jMlpFilter myFilter = new Dl4jMlpFilter();

        // Load our pretrained model (must be done *before* specifying extra transformation layers)
        Dl4JResNet50 zooModel = new Dl4JResNet50();
        myFilter.setZooModelType(zooModel);

        // Concatenate activations from an intermediate convolution layer
        myFilter.addTransformationLayerName("res4a_branch2b");
        // Set the pooling type to average
        myFilter.setPoolingType(PoolingType.AVG);

        // Create our iterator, pointing it to the location of the images
        ImageInstanceIterator imgIter = new ImageInstanceIterator();
        imgIter.setImagesLocation(new File("datasets/nominal/mnist-minimal"));
        // Featurize 16 instances at a time
        imgIter.setTrainBatchSize(16);
        myFilter.setImageInstanceIterator(imgIter);

// Run the filter, using the model as a feature extractor
        myFilter.setInputFormat(instances);
        Instances transformedInstances = Filter.useFilter(instances, myFilter);

// CV our Random Forest classifier on the extracted features
        Evaluation evaluation = new Evaluation(transformedInstances);
        int numFolds = 10;
        evaluation.crossValidateModel(new RandomForest(), transformedInstances, numFolds, new Random(1));
        System.out.println(evaluation.toSummaryString());
    }

    public void filterTest(String[] args) throws Exception {
        Dl4jMlpFilter myFilter = new Dl4jMlpFilter();
        ImageInstanceIterator imgIter = new ImageInstanceIterator();
        imgIter.setImagesLocation(new File("datasets/nominal/mnist-minimal"));
        imgIter.setTrainBatchSize(16);
        myFilter.setImageInstanceIterator(imgIter);
//        myFilter.setTransformationLayerNames(new String[] {"res4a_branch2b"});
        KerasResNet zooModel = new KerasResNet();
        zooModel.setVariation(ResNet.VARIATION.RESNET152V2);
        myFilter.setZooModelType(zooModel);
        Filter.runFilter(myFilter, args);
    }

    public void train(String[] args) throws Exception {
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);
        clf.setNumEpochs(2);

        // Load the arff file
        Instances data = new Instances(new FileReader("E:\\Rhys\\Documents\\Github\\kaggle-competitions\\plant-seedlings\\data\\train\\plant-seedlings-train.arff"));
        data.setClassIndex(data.numAttributes() - 1);

        ImageInstanceIterator imgIter = new ImageInstanceIterator();
        imgIter.setImagesLocation(new File("E:\\Rhys\\Documents\\Github\\kaggle-competitions\\plant-seedlings\\data\\train"));
        imgIter.setTrainBatchSize(16);
        clf.setInstanceIterator(imgIter);

        // Set up the network configuration
        NeuralNetConfiguration nnc = new NeuralNetConfiguration();
        Updater updater = new Adam();
        updater.setLearningRate(0.1);
        nnc.setUpdater(updater);
        clf.setNeuralNetConfiguration(nnc);

        // Set up the pretrained model
        KerasResNet zooModel = new KerasResNet();
        zooModel.setVariation(ResNet.VARIATION.RESNET152V2);
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
}

public class Main {
    public static void main(String[] args) throws Exception {
        new WekaTests().train(args);
    }
}
