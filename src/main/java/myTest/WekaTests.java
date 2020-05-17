package myTest;

import org.deeplearning4j.zoo.ZooModel;
import org.nd4j.shade.protobuf.MapEntry;
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
import weka.dl4j.zoo.keras.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Dl4jMlpFilter;

import java.io.File;
import java.io.FileReader;
import java.util.*;

public class WekaTests {

    private static List<AbstractZooModel> createKerasModelVariations() {
        List<AbstractZooModel> kerasModels = new ArrayList<>();

//        KerasDenseNet denseNet121 = new KerasDenseNet();
//        denseNet121.setVariation(DenseNet.VARIATION.DENSENET121);
//        kerasModels.add(denseNet121);
//
//        KerasDenseNet denseNet169 = new KerasDenseNet();
//        denseNet169.setVariation(DenseNet.VARIATION.DENSENET169);
//        kerasModels.add(denseNet169);
//
//        KerasDenseNet denseNet201 = new KerasDenseNet();
//        denseNet201.setVariation(DenseNet.VARIATION.DENSENET201);
//        kerasModels.add(denseNet201);

//        EfficientNet efficientNetB0 = new EfficientNet();
//        efficientNetB0.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B0);
//        kerasModels.add(efficientNetB0);
//
//        EfficientNet efficientNetB1 = new EfficientNet();
//        efficientNetB1.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B1);
//        kerasModels.add(efficientNetB1);
//
//        EfficientNet efficientNetB2 = new EfficientNet();
//        efficientNetB2.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B2);
//        kerasModels.add(efficientNetB2);
//
//        EfficientNet efficientNetB3 = new EfficientNet();
//        efficientNetB3.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B3);
//        kerasModels.add(efficientNetB3);
//
//        EfficientNet efficientNetB4 = new EfficientNet();
//        efficientNetB4.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B4);
//        kerasModels.add(efficientNetB4);
//
//        EfficientNet efficientNetB5 = new EfficientNet();
//        efficientNetB5.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B5);
//        kerasModels.add(efficientNetB5);
//
//        EfficientNet efficientNetB6 = new EfficientNet();
//        efficientNetB6.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B6);
//        kerasModels.add(efficientNetB6);
//
//        EfficientNet efficientNetB7 = new EfficientNet();
//        efficientNetB7.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B7);
//        kerasModels.add(efficientNetB7);

//        InceptionResNetV2 inceptionResNetV2 = new InceptionResNetV2();
//        inceptionResNetV2.setVariation(InceptionResNetV2.VARIATION.STANDARD);
//        kerasModels.add(inceptionResNetV2);

//        KerasInceptionV3 inceptionV3 = new KerasInceptionV3();
//        inceptionV3.setVariation(InceptionV3.VARIATION.STANDARD);
//        kerasModels.add(inceptionV3);

//        MobileNet mobileNet = new MobileNet();
//        mobileNet.setVariation(MobileNet.VARIATION.V1);
//        kerasModels.add(mobileNet);
//
//        MobileNet mobileNetV2 = new MobileNet();
//        mobileNet.setVariation(MobileNet.VARIATION.V2);
//        kerasModels.add(mobileNetV2);

//        KerasNASNet nasNetMobile = new KerasNASNet();
//        nasNetMobile.setVariation(weka.dl4j.zoo.keras.NASNet.VARIATION.MOBILE);
//        kerasModels.add(nasNetMobile);

//        KerasNASNet nasNetLarge = new KerasNASNet();
//        nasNetLarge.setVariation(weka.dl4j.zoo.keras.NASNet.VARIATION.LARGE);
//        kerasModels.add(nasNetLarge);

        KerasResNet resNet50 = new KerasResNet();
        resNet50.setVariation(ResNet.VARIATION.RESNET50);
        kerasModels.add(resNet50);

        KerasResNet resNet50V2 = new KerasResNet();
        resNet50V2.setVariation(ResNet.VARIATION.RESNET50V2);
        kerasModels.add(resNet50V2);

        KerasResNet resNet101 = new KerasResNet();
        resNet101.setVariation(ResNet.VARIATION.RESNET101);
        kerasModels.add(resNet101);

        KerasResNet resNet101V2 = new KerasResNet();
        resNet101V2.setVariation(ResNet.VARIATION.RESNET101V2);
        kerasModels.add(resNet101V2);

        KerasResNet resNet152 = new KerasResNet();
        resNet152.setVariation(ResNet.VARIATION.RESNET152);
        kerasModels.add(resNet152);

        KerasResNet resNet152V2 = new KerasResNet();
        resNet152V2.setVariation(ResNet.VARIATION.RESNET152V2);
        kerasModels.add(resNet152V2);

//        KerasVGG vgg16 = new KerasVGG();
//        vgg16.setVariation(VGG.VARIATION.VGG16);
//        kerasModels.add(vgg16);
//
//        KerasVGG vgg19 = new KerasVGG();
//        vgg19.setVariation(VGG.VARIATION.VGG19);
//        kerasModels.add(vgg19);
//
//        KerasXception xception = new KerasXception();
//        xception.setVariation(weka.dl4j.zoo.keras.Xception.VARIATION.STANDARD);
//        kerasModels.add(xception);

        return kerasModels;
    }

    public static List<AbstractZooModel> createModelsToDownload() {
        List<AbstractZooModel> models = new ArrayList<>();
        models.add(new Dl4jDarknet19());
        models.add(new Dl4jLeNet());
        models.add(new Dl4JResNet50());
        models.add(new Dl4jSqueezeNet());

        Dl4jVGG vgg16 = new Dl4jVGG();
        vgg16.setVariation(VGG.VARIATION.VGG16);

        Dl4jVGG vgg19 = new Dl4jVGG();
        vgg19.setVariation(VGG.VARIATION.VGG19  );

        models.add(vgg16);
        models.add(vgg16);
        models.add(new Dl4jXception());

        return createKerasModelVariations();
//        models.addAll(kerasModels);

//        return models;
    }

    public void filterExampleReflection(String[] args) throws Exception {
// Load all packages so that Dl4jMlpFilter class can be found using forName("weka.filters.unsupervised.attribute.Dl4jMlpFilter")
        weka.core.WekaPackageManager.loadPackages(true);

// Load the dataset
        weka.core.Instances instances = new weka.core.Instances(new FileReader("datasets/nominal/mnist.meta.minimal.arff"));
        instances.setClassIndex(1);
        String[] filterOptions = weka.core.Utils.splitOptions("-iterator \".ImageInstanceIterator -imagesLocation datasets/nominal/mnist-minimal -bs 12\" -poolingType AVG -layer-extract \"weka.dl4j.layers.DenseLayer -name res4a_branch2b\" -layer-extract \"weka.dl4j.layers.DenseLayer -name flatten_1\" -isZoo true -zooModel \".Dl4JResNet50\"");
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
        evaluation.crossValidateModel(new weka.classifiers.functions.SMO(), transformedInstances, numFolds, new Random(1));
        System.out.println(evaluation.toSummaryString());
        System.out.println(evaluation.toMatrixString());
    }

    public void filterExample(String[] args) throws Exception {
        // Load the dataset
        Instances instances = new Instances(new FileReader("datasets/nominal/mnist.meta.minimal.arff"));
        instances.setClassIndex(1);
        Dl4jMlpFilter myFilter = new Dl4jMlpFilter();

// Create our iterator, pointing it to the location of the images
        ImageInstanceIterator imgIter = new ImageInstanceIterator();
        imgIter.setImagesLocation(new File("datasets/nominal/mnist-minimal"));
// Set batch size to thread count if using CPU
        imgIter.setTrainBatchSize(12);
        myFilter.setImageInstanceIterator(imgIter);

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
        List<AbstractZooModel> pretrainedModels = createModelsToDownload();
        List<AbstractZooModel> failedModels = new ArrayList<>();
        int successCount = 0;
        int originalNum = pretrainedModels.size();
        while (pretrainedModels.size() > 0) {
            Dl4jMlpFilter myFilter = new Dl4jMlpFilter();
            ImageInstanceIterator imgIter = new ImageInstanceIterator();
            imgIter.setImagesLocation(new File("datasets/nominal/mnist-minimal"));
            imgIter.setTrainBatchSize(16);
            myFilter.setImageInstanceIterator(imgIter);

            AbstractZooModel thisModel = pretrainedModels.get(0);
            myFilter.setZooModelType(thisModel);

            weka.core.Instances instances = new weka.core.Instances(new FileReader(args[1]));
            instances.setClassIndex(1);

            myFilter.setInputFormat(instances);
            try {
                Instances newInst = Filter.useFilter(instances, myFilter);
                System.out.println(newInst.size());
                successCount++;
            } catch (Exception e) {
                e.printStackTrace();
                failedModels.add(thisModel);
            }
            pretrainedModels.remove(0);
            System.gc();
//            Filter.runFilter(myFilter, args);
        }
        System.out.println("Failed models: ");
        for (AbstractZooModel zooModel : failedModels) {
            System.out.println(zooModel.getClass());
        }
        System.out.println(successCount);
        System.out.println(originalNum);
    }

    public void filterTest2(String[] args) throws Exception {
            Dl4jMlpFilter myFilter = new Dl4jMlpFilter();
            ImageInstanceIterator imgIter = new ImageInstanceIterator();
            imgIter.setImagesLocation(new File("datasets/nominal/mnist-minimal"));
            imgIter.setTrainBatchSize(16);
            imgIter.setHeight(224);
            imgIter.setWidth(224);
            imgIter.setNumChannels(3);
            myFilter.setImageInstanceIterator(imgIter);

            Dl4JResNet50 thisModel = new Dl4JResNet50();
//            thisModel.setPretrainedType(PretrainedType.NONE);
//            thisModel.setVariation(NASNet.VARIATION.LARGE);
            myFilter.setZooModelType(thisModel);

//            weka.core.Instances instances = new weka.core.Instances(new FileReader(args[1]));
//            instances.setClassIndex(1);
//
//            myFilter.setInputFormat(instances);

            Filter.runFilter(myFilter, args);
    }

    public void train(String[] args) throws Exception {
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);
        clf.setNumEpochs(1);

        // Load the arff file
        Instances data = new Instances(new FileReader("datasets/nominal/mnist.meta.tiny.arff"));
        data.setClassIndex(data.numAttributes() - 1);

        ImageInstanceIterator imgIter = new ImageInstanceIterator();
        imgIter.setImagesLocation(new File("datasets/nominal/mnist-minimal"));
        imgIter.setTrainBatchSize(16);
        imgIter.setWidth(224);
        imgIter.setHeight(224);
        imgIter.setNumChannels(3);
        clf.setInstanceIterator(imgIter);

        // Set up the pretrained model
        Dl4jDarknet19 zooModel = new Dl4jDarknet19();
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
        new WekaTests().filterTest2(args);
    }
}
