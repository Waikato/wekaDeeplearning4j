package weka.examples;

import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.Instances;
import weka.core.converters.ImageDirectoryLoader;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.zoo.KerasEfficientNet;
import weka.dl4j.zoo.keras.EfficientNet;

import java.io.File;

public class WekaDeeplearning4jExamples {

    public static void main(String[] args) throws Exception {
        dl4jResnet50();
    }

    private static void dl4jResnet50() throws Exception {
        String folderPath = "src/test/resources/nominal/plant-seedlings-small";
        ImageDirectoryLoader loader = new ImageDirectoryLoader();
        loader.setInputDirectory(new File(folderPath));
        Instances inst = loader.getDataSet();
        inst.setClassIndex(1);

        Dl4jMlpClassifier classifier = new Dl4jMlpClassifier();
        classifier.setNumEpochs(3);

        KerasEfficientNet kerasEfficientNet = new KerasEfficientNet();
        kerasEfficientNet.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B1);
        classifier.setZooModel(kerasEfficientNet);

        ImageInstanceIterator iterator = new ImageInstanceIterator();
        iterator.setImagesLocation(new File(folderPath));

        classifier.setInstanceIterator(iterator);

        classifier.buildClassifier(inst);
    }
}
