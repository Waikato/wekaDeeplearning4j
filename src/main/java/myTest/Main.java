package myTest;

import org.deeplearning4j.zoo.PretrainedType;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.Instances;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.activations.ActivationReLU;
import weka.dl4j.activations.ActivationSoftmax;
import weka.dl4j.earlystopping.EarlyStopping;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.layers.DenseLayer;
import weka.dl4j.layers.OutputLayer;
import weka.dl4j.updater.Adam;
import weka.dl4j.zoo.ResNet50;
import weka.dl4j.zoo.VGG16;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.File;
import java.io.FileReader;
import java.nio.file.Paths;
import java.util.Random;

class IrisNN {
    public IrisNN() {}

    public void run() throws Exception {
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        DenseLayer denseLayer = new DenseLayer();
        denseLayer.setNOut(32);
        denseLayer.setActivationFunction(new ActivationReLU());

        // Define the output layer
        OutputLayer outputLayer = new OutputLayer();
        outputLayer.setActivationFunction(new ActivationSoftmax());

        clf.setLayers(denseLayer, outputLayer);

        clf.setNumEpochs(10);

        String irisPath = "/home/rhys/Documents/git/wekaDeeplearning4j/datasets/nominal/iris.arff";
        Instances inst = new Instances(new FileReader(irisPath));
        inst.setClassIndex(inst.numAttributes() - 1);
        Evaluation ev = new Evaluation(inst);
        ev.crossValidateModel(clf, inst, 3, new Random(0));
        System.out.println(ev.toSummaryString());
    }
}

class ResnetTest {
    public ResnetTest() {}

    public void run() throws Exception {
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);
        clf.setNumEpochs(50);
        clf.setZooModel(new VGG16());

        // Load the arff file
        Instances data = new Instances(new FileReader("datasets/nominal/iris_reloaded/iris_reloaded.arff"));

        data.setClassIndex(data.numAttributes() - 1);

        ImageInstanceIterator imgIter = new ImageInstanceIterator();
        imgIter.setImagesLocation(new File("datasets/nominal/iris_reloaded"));
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

        RemovePercentage removePercentage = new RemovePercentage();
        removePercentage.setPercentage(10);

        randData.stratify(10);

        Instances train = randData.trainCV(10, 0);
        clf.buildClassifier(train);


        Instances test = randData.testCV(10, 0);
        Evaluation eval = new Evaluation(test);
        eval.evaluateModel(clf, test);

        System.out.println("% Correct = " + eval.toSummaryString());
    }
}

public class Main {
    public static void main(String[] args) throws Exception {
        ResnetTest test = new ResnetTest();
        test.run();
    }
}
