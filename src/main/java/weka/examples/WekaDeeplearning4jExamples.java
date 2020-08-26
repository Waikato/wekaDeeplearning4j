package weka.examples;

import org.deeplearning4j.nn.graph.ComputationGraph;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.classifiers.functions.dl4j.Utils;
import weka.core.Instances;
import weka.core.converters.ImageDirectoryLoader;
import weka.dl4j.interpretability.AbstractSaliencyMapWrapper;
import weka.dl4j.interpretability.ScoreCAM;
import weka.dl4j.interpretability.WekaScoreCAM;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.inference.*;
import weka.dl4j.zoo.*;
import weka.dl4j.zoo.keras.EfficientNet;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Dl4jMlpFilter;

import java.io.File;
import java.io.FileReader;
import java.util.Random;

public class WekaDeeplearning4jExamples {

    public static void main(String[] args) throws Exception {
        inference();
    }

    public static void commandLineProgressTest() throws Exception {
        String folderPath = "datasets/nominal/mnist-minimal";
        Instances inst = new Instances(new FileReader("datasets/nominal/mnist.meta.tiny.arff"));
        inst.setClassIndex(1);

        Dl4jMlpFilter filter = new Dl4jMlpFilter();

        ImageInstanceIterator iterator = new ImageInstanceIterator();
        iterator.setTrainBatchSize(4);
        iterator.setImagesLocation(new File(folderPath));

        KerasEfficientNet kerasEfficientNet = new KerasEfficientNet();
        kerasEfficientNet.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B1);
        filter.setZooModelType(kerasEfficientNet);

        filter.setUseDefaultFeatureLayer(false);
        filter.setTransformationLayerNames(new String[] {
                "top_dropout",
                "block7b_se_expand"
        });

        filter.setInstanceIterator(iterator);
        filter.setInputFormat(inst);

        Instances filteredInstances = Filter.useFilter(inst, filter);
        System.out.println(filteredInstances);
    }

    private static void scoreCamTest() {
        Dl4jResNet50 pretrainedModel = new Dl4jResNet50();
//        pretrainedModel.setVariation(ResNet.VARIATION.RESNET101V2);
        ComputationGraph computationGraph = pretrainedModel.getDefaultGraph();

        ScoreCAM scoreCAM = new ScoreCAM();
        scoreCAM.setBatchSize(8);
        scoreCAM.setComputationGraph(computationGraph);
        scoreCAM.setImageChannelsLast(pretrainedModel.getChannelsLast());
        scoreCAM.setModelInputShape(Utils.decodeCNNShape(pretrainedModel.getShape()[0]));
        scoreCAM.setImagePreProcessingScaler(pretrainedModel.getImagePreprocessingScaler());
        scoreCAM.generateForImage(new File("src/test/resources/images/dog.jpg"));
    }

    private static void filter() throws Exception {
        String folderPath = "src/test/resources/nominal/plant-seedlings-small";
        ImageDirectoryLoader loader = new ImageDirectoryLoader();
        loader.setInputDirectory(new File(folderPath));
        Instances inst = loader.getDataSet();
        inst.setClassIndex(1);

        Dl4jMlpFilter filter = new Dl4jMlpFilter();

        ImageInstanceIterator iterator = new ImageInstanceIterator();
        iterator.setImagesLocation(new File(folderPath));

        KerasEfficientNet kerasEfficientNet = new KerasEfficientNet();
        kerasEfficientNet.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B1);
        filter.setZooModelType(kerasEfficientNet);

        filter.setUseDefaultFeatureLayer(false);
        filter.setTransformationLayerNames(new String[] {
                "top_dropout",
                "block7b_se_expand"
        });

        filter.setInstanceIterator(iterator);
        filter.setInputFormat(inst);

        Instances filteredInstances = Filter.useFilter(inst, filter);
        System.out.println(filteredInstances);
    }

    private static void train() throws Exception {
        String folderPath = "src/test/resources/nominal/plant-seedlings-small";
        ImageDirectoryLoader loader = new ImageDirectoryLoader();
        loader.setInputDirectory(new File(folderPath));
        Instances inst = loader.getDataSet();
        inst.setClassIndex(1);

        Dl4jMlpClassifier classifier = new Dl4jMlpClassifier();
        classifier.setNumEpochs(3);

        KerasDenseNet kerasEfficientNet = new KerasDenseNet();
//        kerasEfficientNet.setVariation(ResNet.VARIATION.RESNET50V2);
//        kerasEfficientNet.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B1);
        classifier.setZooModel(kerasEfficientNet);

        ImageInstanceIterator iterator = new ImageInstanceIterator();
        iterator.setImagesLocation(new File(folderPath));

        classifier.setInstanceIterator(iterator);

        // Stratify and split the data
        Random rand = new Random(0);
        inst.randomize(rand);
        inst.stratify(5);
        Instances train = inst.trainCV(5, 0);
        Instances test = inst.testCV(5, 0);

        // Build the classifier on the training data
        classifier.buildClassifier(train);

        // Evaluate the model on test data
        Evaluation eval = new Evaluation(test);
        eval.evaluateModel(classifier, test);

        // Output some summary statistics
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());
    }

    public static void inference() throws Exception {
        Dl4jCNNExplorer explorer = new Dl4jCNNExplorer();

        Dl4jResNet50 zooModel = new Dl4jResNet50();
//        zooModel.setVariation(ResNet.VARIATION.RESNET152V2);
//        zooModel.setVariation(VGG.VARIATION.VGG16);
//        zooModel.setPretrainedType(PretrainedType.VGGFACE);
        explorer.setZooModelType(zooModel);

        ModelOutputDecoder decoder = new ModelOutputDecoder();
        decoder.setBuiltInClassMap(ModelOutputDecoder.ClassmapType.IMAGENET);
        explorer.setModelOutputDecoder(decoder);

        explorer.setGenerateSaliencyMap(true);
        AbstractSaliencyMapWrapper wrapper = new WekaScoreCAM();
        wrapper.setBatchSize(8);
        explorer.setSaliencyMapGenerator(wrapper);

        explorer.init();
        explorer.makePrediction(new File("src/test/resources/images/dog.jpg"));

        System.out.println(explorer.getCurrentPredictions().toSummaryString());
    }
}
