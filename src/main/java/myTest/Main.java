package myTest;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.Instances;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.listener.EpochListener;
import weka.dl4j.updater.Adam;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Dl4jMlpFilter;

import java.io.File;
import java.io.FileReader;
import java.util.Random;

class ResnetTest {
    public ResnetTest() {}

    public void filterTest(String[] args) {
        Dl4jMlpFilter myFilter = new Dl4jMlpFilter();
        ImageInstanceIterator imgIter = new ImageInstanceIterator();
        imgIter.setImagesLocation(new File("datasets/nominal/mnist-minimal"));
        myFilter.setImageInstanceIterator(imgIter);
        Filter.runFilter(myFilter, args);
    }

    public void train() throws Exception {
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);
        clf.setNumEpochs(10);
//        clf.setZooModel(new ResNet50(PretrainedType.IMAGENET));
//        clf.setZooModel(new GenericKerasModel(
//                new ClassPathResource("vgg16.h5").getFile().getPath(),
//                new ClassPathResource("resnet50.json").getFile().getPath()));

        final EpochListener epochListener = new EpochListener();
        epochListener.setN(1);
        clf.setIterationListener(epochListener);

        // Load the arff file
        Instances data = new Instances(new FileReader("datasets/nominal/mnist.meta.minimal.arff"));

        data.setClassIndex(data.numAttributes() - 1);

        ImageInstanceIterator imgIter = new ImageInstanceIterator();
        imgIter.setImagesLocation(new File("datasets/nominal/mnist-minimal"));
        imgIter.setHeight(224);
        imgIter.setWidth(224);
        imgIter.setNumChannels(3);
        imgIter.setTrainBatchSize(16);
        clf.setInstanceIterator(imgIter);

        // Set up the network configuration
        NeuralNetConfiguration nnc = new NeuralNetConfiguration();
        nnc.setUpdater(new Adam());
        clf.setNeuralNetConfiguration(nnc);

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
        new ResnetTest().filterTest(args);
    }
}
