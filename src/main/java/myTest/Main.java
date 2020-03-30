package myTest;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.Instances;
import weka.dl4j.PretrainedType;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.zoo.*;
import weka.dl4j.zoo.keras.*;
import weka.dl4j.zoo.keras.NASNet;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Dl4jMlpFilter;

import java.io.File;
import java.io.FileReader;
import java.util.Random;

class WekaTests {
    public WekaTests() {}

    public void filterTest(String[] args) throws Exception {
        Dl4jMlpFilter myFilter = new Dl4jMlpFilter();
        ImageInstanceIterator imgIter = new ImageInstanceIterator();
        imgIter.setImagesLocation(new File("datasets/nominal/mnist-minimal"));
        imgIter.setTrainBatchSize(16);
        imgIter.setNumChannels(3); // TODO auto set for keras model
        myFilter.setImageInstanceIterator(imgIter);
        KerasNASNet zooModel = new KerasNASNet();
        zooModel.setVariation(NASNet.VARIATION.LARGE);
//        zooModel.setPretrainedType(PretrainedType.VGGFACE);
//        zooModel.setVariation(InceptionResNetV2.VARIATION.STANDARD);
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

        ResNet50 zooModel = new ResNet50();
        zooModel.setPretrainedType(PretrainedType.IMAGENET);
//        zooModel.setKerasH5File(new ClassPathResource("mobilenetv2.h5").getFile().getPath());
//        zooModel.setKerasJsonFile(new ClassPathResource("mobilenetv2.json").getFile().getPath());
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
        new WekaTests().filterTest(args);
    }
}
