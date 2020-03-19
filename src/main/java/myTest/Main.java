package myTest;

import org.nd4j.linalg.io.ClassPathResource;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.Instances;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.PretrainedType;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.listener.EpochListener;
import weka.dl4j.updater.Adam;
import weka.dl4j.updater.RmsProp;
import weka.dl4j.updater.Updater;
import weka.dl4j.zoo.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Dl4jMlpFilter;

import java.io.File;
import java.io.FileReader;
import java.util.Random;

class WekaTests {
    public WekaTests() {}

    public void filterTest(String[] args) {
        Dl4jMlpFilter myFilter = new Dl4jMlpFilter();
        ImageInstanceIterator imgIter = new ImageInstanceIterator();
        imgIter.setImagesLocation(new File("/home/rhys/Downloads/test"));
        imgIter.setTrainBatchSize(16);
        myFilter.setImageInstanceIterator(imgIter);
        AbstractZooModel zooModel = new ResNet50();
//        zooModel.setPretrainedType(PretrainedType.CIFAR10);
        myFilter.setZooModelType(zooModel);
        Filter.runFilter(myFilter, args);
    }

    public void train(String[] args) throws Exception {
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);
        clf.setNumEpochs(10);

        // Load the arff file
        Instances data = new Instances(new FileReader("datasets/nominal/mnist.meta.minimal.arff"));

        data.setClassIndex(data.numAttributes() - 1);

        ImageInstanceIterator imgIter = new ImageInstanceIterator();
        imgIter.setImagesLocation(new File("datasets/nominal/mnist-minimal"));
        imgIter.setTrainBatchSize(16);
        clf.setInstanceIterator(imgIter);

        // Set up the network configuration
//        NeuralNetConfiguration nnc = new NeuralNetConfiguration();
//        Updater updater = new Adam();
//        updater.setLearningRate(0.1);
//        adam.setLearningRate(0.1);
//        nnc.setUpdater(updater);
//        clf.setNeuralNetConfiguration(nnc);

        GenericKerasModel zooModel = new GenericKerasModel();
        zooModel.setKerasH5File(new ClassPathResource("mobilenetv2.h5").getFile().getPath());
        zooModel.setKerasJsonFile(new ClassPathResource("mobilenetv2.json").getFile().getPath());
        clf.setZooModel(zooModel);

        Random rand = new Random(0);
        Instances randData = new Instances(data);
        randData.randomize(rand);

        randData.stratify(3);

        Instances train = randData.trainCV(3, 0);
        clf.buildClassifier(train);

        Instances test = randData.testCV(3, 0);
        Evaluation eval = new Evaluation(test);
        eval.evaluateModel(clf, test);

        System.out.println(eval.toSummaryString());
    }
}

public class Main {
    public static void main(String[] args) throws Exception {
//        ResnetTest test = new ResnetTest();
//        test.train();
        new WekaTests().train(args);
    }
}
