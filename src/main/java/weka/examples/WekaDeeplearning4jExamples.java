package weka.examples;

import com.sun.jna.platform.win32.OaIdl;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.enums.ImageResizeMethod;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.Normalizer;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.ops.NDImage;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.classifiers.functions.dl4j.Utils;
import weka.core.Instances;
import weka.core.converters.ImageDirectoryLoader;
import weka.dl4j.PretrainedType;
import weka.dl4j.interpretability.ScoreCAM;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.playground.*;
import weka.dl4j.zoo.*;
import weka.dl4j.zoo.keras.EfficientNet;
import weka.dl4j.zoo.keras.ResNet;
import weka.dl4j.zoo.keras.VGG;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Dl4jMlpFilter;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

public class WekaDeeplearning4jExamples {

    public static void main(String[] args) throws Exception {
        scoreCamTest2();
    }

    private static void scoreCamTest2() {
        Dl4jResNet50 resNet50 = new Dl4jResNet50();
        ComputationGraph computationGraph = resNet50.init(2, 1, resNet50.getShape()[0], true);

        ScoreCAM scoreCAM = new ScoreCAM(computationGraph, "res5c_branch2c");
        scoreCAM.generateForImage(new File("src/test/resources/images/dog.jpg"), 235);
    }

    private static void printArr(long[] arr) {
        System.out.println(Arrays.toString(arr));
    }

    private static void scoreCamForResizeMethod(ImageResizeMethod method) throws Exception {
        int targetID = 817;
        // Load the VGG16 model
        ComputationGraph vgg16Model = ComputationGraph.load(new File("/home/rhys/.deeplearning4j/models/resnet50/resnet50_dl4j_inference.v3.zip"), false);
        TransferLearningHelper transferLearningHelper = new TransferLearningHelper(vgg16Model.clone(), "res5c_branch2c");

        // Load the image
        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        INDArray imageArr = loader.asMatrix(new File("src/test/resources/images/car.jpg"));

        DataSet imageDataset = new DataSet(imageArr, Nd4j.zeros(1));

        // Get the activation maps
        DataSet rawActivations = transferLearningHelper.featurize(imageDataset);
        // Must be channels last for the imageResize method
        INDArray rawActivationsChannelsLast = rawActivations.getFeatures().permute(0, 2, 3, 1);
        printArr(rawActivationsChannelsLast.shape());


        // Create the new size array
        INDArray newSize = Nd4j.create(new int[] {224, 224}, new long[] {2}, DataType.INT32);

        // Upsample the activations to match original image size
        NDImage ndImage = new NDImage();
        INDArray upsampledActivations = ndImage.imageResize(rawActivationsChannelsLast, newSize, method); // TODO try BiLinear

        // Drop the 1 from [1, 224, 224, 512]
        upsampledActivations = Nd4j.squeeze(upsampledActivations, 0);

        // Reshape back to [C, H, W] (easier to iterate over)
        upsampledActivations = upsampledActivations.permute(2, 0, 1);

        // Normalize each of the 512 activation maps
        int numActivationMaps = (int) upsampledActivations.shape()[0];
        for (int i = 0; i < numActivationMaps; i++) {
            INDArray tmpActivationMap = upsampledActivations.get(NDArrayIndex.point(i));
            double maxVal = tmpActivationMap.maxNumber().doubleValue();
            double minVal = tmpActivationMap.minNumber().doubleValue();
            double fudgeVal = 1e-5;
            double divisor = (maxVal - minVal) + fudgeVal;

            tmpActivationMap.divi(divisor);
        }
        // Renaming to make the process easier to follow
        INDArray normalisedActivations = upsampledActivations;

        // [1, 3, 224, 224] -> [3, 224, 224] - remove the minibatch dimension
        imageArr = Nd4j.squeeze(imageArr, 0);
        System.out.println("Image shape: " + Arrays.toString(imageArr.shape()));

        INDArray allModelPasses = Nd4j.zeros(numActivationMaps, 1000);

        // Create the 512 masked images -
        // Multiply each normalized activation map with the image
        for (int i = 0; i < numActivationMaps; i++) {
            INDArray iActivationMap = normalisedActivations.get(NDArrayIndex.point(i));
            // [224, 224] -> [1, 224, 224] (is then broadcasted in the multiply method)
            iActivationMap = iActivationMap.reshape(1, 224, 224);

            // [3, 224, 224] . [1, 224, 224] - actually create the masked image
            INDArray multiplied = imageArr.mul(iActivationMap);

//            System.out.println("Multiplied max value is " + multiplied.maxNumber().doubleValue());
//            BufferedImage img = imageFromINDArray(multiplied);
////            ImageIO.write(img, "png", new File(String.format("activationMaps/%d_%s.png", i, method.toString())));
//            ImageIO.write(img, "png", new File(String.format("before.png", i, method.toString())));

            // Needs the minibatch added back in for prediction
            multiplied = multiplied.reshape(1, 3, 224, 224);

//            img = imageFromINDArray(multiplied);
////            ImageIO.write(img, "png", new File(String.format("activationMaps/%d_%s.png", i, method.toString())));
//            ImageIO.write(img, "png", new File("after.png"));
//
//            System.exit(0);

            INDArray output = vgg16Model.outputSingle(multiplied);

            // Save this model's prediction scores
            INDArrayIndex[] index = new INDArrayIndex[] { NDArrayIndex.point(i)};
            allModelPasses.put(index, output);
        }

        INDArray weights = Nd4j.zeros(numActivationMaps);
        for (int i = 0; i < numActivationMaps; i++) {
            INDArray rowVecForMap = allModelPasses.getRow(i);
            double classProbVal = rowVecForMap.getDouble(targetID);
            weights.putScalar(i, classProbVal);
//            INDArray maxIndex = rowVecForMap.argMax(0);
//            System.out.println(Arrays.toString(maxIndex.shape()));
        }

        // Add dimensions to the weights for the multiplication
        weights = weights.reshape(numActivationMaps, 1, 1);

        INDArray finalMap = normalisedActivations.mul(weights);

        // Sum all maps to get one 224x224 map - [224, 224]
        INDArray summed = finalMap.sum(0);
        // Perform pixel-wise RELU
        INDArray finalRELU = Transforms.relu(summed);
        // Scale the map to between 0 and 1 (so it can be multiplied on the image)
        double currMax = finalRELU.maxNumber().doubleValue();
        double currMin = finalRELU.minNumber().doubleValue();
//        double fudgeVal = 1e-5;
        System.out.println(String.format("Prev max: %.4f, prev min: %.4f", currMax, currMin));

        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0.0, 1.0);
        normalizer.fit(new DataSet(finalRELU, Nd4j.zeros(1)));
        normalizer.transform(finalRELU);

        double newMax = finalRELU.maxNumber().doubleValue();
        double newMin = finalRELU.minNumber().doubleValue();
        System.out.println(String.format("new max: %.4f, new min: %.4f", newMax, newMin));

        // Scale from [0, 1] to [0, 255] (for displaying as an image)
//        finalRELU.muli(255);
//        System.out.println("Final relu max number is " + finalRELU.maxNumber().doubleValue());

        INDArray reshapedForSaving = finalRELU.reshape(1, 224, 224);

        saveNDArray(imageArr, "1_rawImage.png");
        imageArr.muli(reshapedForSaving);
        saveNDArray(imageArr, "1_masked.png");
        // Multiply the
//        reshapedForSaving = reshapedForSaving.broadcast(3, 224, 224);
//        BufferedImage img = imageFromINDArray(reshapedForSaving);
//        ImageIO.write(img, "png", new File("finalMap.png"));
    }

    private static void saveNDArray(INDArray array, String filename) throws Exception{
        BufferedImage img = imageFromINDArray(array);
        ImageIO.write(img, "png", new File(filename));
    }

    private static void scoreCamTest() throws Exception {
//        for (ImageResizeMethod method : ImageResizeMethod.values()) {
//            try {
//                System.out.println("Running - " + method);
//                scoreCamForResizeMethod(method);
//            } catch (Exception ex) {
//                System.out.println("Couldn't do " + method);
//            }
//        }
        scoreCamForResizeMethod(ImageResizeMethod.ResizeBicubic);
    }

    /**
     * Takes an INDArray containing an image loaded using the native image loader
     * libraries associated with DL4J, and converts it into a BufferedImage.
     * The INDArray contains the color values split up across three channels (RGB)
     * and in the integer range 0-255.
     *
     * @param array INDArray containing an image
     * @return BufferedImage
     */
    private static BufferedImage imageFromINDArray(INDArray array) {
        long[] shape = array.shape();

        boolean is4d = false;

        if (shape.length == 4) {
            is4d = true;
            System.out.println("Map is 4d");
        }

        long height = shape[1];
        long width = shape[2];

        if (is4d) {
            height = shape[2];
            width = shape[3];
        }

        BufferedImage image = new BufferedImage((int) width, (int) height, BufferedImage.TYPE_INT_RGB);
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int red, green, blue;

                if (is4d) {
                    red = array.getInt(0, 2, y, x);
                    green = array.getInt(0, 1, y, x);
                    blue = array.getInt(0, 0, y, x);
                } else {
                    red = array.getInt(2, y, x);
                    green = array.getInt(1, y, x);
                    blue = array.getInt(0, y, x);
                }

                //handle out of bounds pixel values
                red = Math.min(red, 255);
                green = Math.min(green, 255);
                blue = Math.min(blue, 255);

                red = Math.max(red, 0);
                green = Math.max(green, 0);
                blue = Math.max(blue, 0);
                image.setRGB(x, y, new Color(red, green, blue).getRGB());
            }
        }
        return image;
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

    public static void playground() throws Exception {
        Dl4jCNNExplorer explorer = new Dl4jCNNExplorer();

        Dl4jVGG zooModel = new Dl4jVGG();
//        zooModel.setVariation(ResNet.VARIATION.RESNET152V2);
//        zooModel.setVariation(VGG.VARIATION.VGG16);
//        zooModel.setPretrainedType(PretrainedType.VGGFACE);
        explorer.setZooModelType(zooModel);

        ModelOutputDecoder decoder = new ModelOutputDecoder();
        decoder.setBuiltInClassMap(ModelOutputDecoder.ClassmapType.IMAGENET);
        explorer.setModelOutputDecoder(decoder);

        explorer.init();
        explorer.makePrediction(new File("src/test/resources/images/dog.jpg"));

        System.out.println(explorer.getCurrentPredictions().toSummaryString());
    }
}
