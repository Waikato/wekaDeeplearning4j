package weka.dl4j.interpretability;

import lombok.extern.log4j.Log4j2;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.convolution.ConvolutionLayer;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.nd4j.enums.ImageResizeMethod;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.ops.NDImage;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import weka.classifiers.functions.dl4j.Utils;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

// TODO document
@Log4j2
public class ScoreCAM extends AbstractSaliencyMapGenerator {

    protected InputType.InputTypeConvolutional modelInputShape;

    @Override
    public void generateForImage(String inputImagePath, String outputImagePath) {
        // Set up the model and image
        String activationMapLayer = getActivationMapLayer();

        TransferLearningHelper transferLearningHelper = new TransferLearningHelper(
                getComputationGraph().clone(), activationMapLayer);

        File imageFile = new File(inputImagePath);

        INDArray originalImage = loadImage(imageFile);
        // Preprocess the image if the model requires it
        INDArray preprocessedImage = preprocessImage(originalImage);

        calculateTargetClassID(preprocessedImage);
        // Get the original set of activation maps by taking the activations
        // from the final convolution layer when running the image through the model
        INDArray rawActivations = getActivationsForImage(transferLearningHelper, preprocessedImage);
        // Upsample the activations to match the original image size
        INDArray upsampledActivations = upsampleActivations(rawActivations);
        // Normalise them between 0 and 1 (so they can be multiplied with the images)
        INDArray normalisedActivations = normalizeActivations(upsampledActivations);
        // Create the set of masked images by multiplying each (upsampled, normalized) activation map with the original image
        INDArray maskedImages = createMaskedImages(normalisedActivations, preprocessedImage);
        // Get the softmax score for the target class ID when running the masked images through the model
        INDArray targetClassWeights = predictTargetClassWeights(maskedImages);
        // Weight each activation map using the previously acquired softmax scores
        INDArray weightedActivationMaps = applyActivationMapWeights(normalisedActivations, targetClassWeights);
        // Sum the activation maps into one, and normalise the values to between [0, 1]
        INDArray postprocessedActivations = postprocessActivations(weightedActivationMaps);

        saveResults(originalImage, postprocessedActivations, imageFile.getName() + "_processed");
    }

    private INDArray preprocessImage(INDArray imageArr) {
        ImagePreProcessingScaler scaler = getImagePreProcessingScaler();

        if (scaler == null) {
            log.info("No image preprocessing required");
            return imageArr;
        }

        INDArray preprocessed = imageArr.dup();

        log.info("Applying image preprocessing...");
        scaler.transform(preprocessed);
        return preprocessed;
    }

    private void calculateTargetClassID(INDArray imageArr) {
        if (getTargetClassID() != -1) {
            // Target class has already been set, don't calculate it
            return;
        }
        // Otherwise run the model on the image, and choose argmax as the target class
        INDArray output = getComputationGraph().outputSingle(imageArr);
        int argMax = output.argMax(1).getNumber(0).intValue();
        setTargetClassID(argMax);
    }

    /**
     * Gets the activation map layer to use as the image masks - this is taken as the final convolution layer
     * @return
     */
    private String getActivationMapLayer() {
        Layer[] layers = getComputationGraph().getLayers();
        Layer[] convLayers = Arrays.stream(layers).filter(x -> x instanceof ConvolutionLayer).toArray(Layer[]::new);
        int count = convLayers.length;
        return convLayers[count - 1].getConfig().getLayerName();
    }

    private void saveResults(INDArray imageArr, INDArray saliencyMap, String filename) {
        Utils.saveNDArray(imageArr, filename);
        INDArray masked = imageArr.mul(saliencyMap);
        Utils.saveNDArray(masked, filename + "_masked");
    }

    private INDArray postprocessActivations(INDArray weightedActivationMaps) {
        // Sum all maps to get one 224x224 map - [numActivationMaps, 224, 224] -> [224, 224]
        INDArray summed = weightedActivationMaps.sum(0);
        // Perform pixel-wise RELU
        INDArray reluActivations = Transforms.relu(summed);

        postNormaliseMap(reluActivations);

        // Reshape for saving
        return reluActivations.reshape(1, modelInputShape.getHeight(), modelInputShape.getWidth());
    }

    private void postNormaliseMap(INDArray activationMap) {
        // Perform any normalizing of the activation maps
        // Scale the map to between 0 and 1 (so it can be multiplied on the image)
        double currMax = activationMap.maxNumber().doubleValue();
        double currMin = activationMap.minNumber().doubleValue();
        System.out.println(String.format("Prev max: %.4f, prev min: %.4f", currMax, currMin));

        activationMap.subi(currMin);
        activationMap.divi(currMax - currMin);

        double newMax = activationMap.maxNumber().doubleValue();
        double newMin = activationMap.minNumber().doubleValue();
        System.out.println(String.format("new max: %.4f, new min: %.4f", newMax, newMin));
    }

    private INDArray applyActivationMapWeights(INDArray normalisedActivations, INDArray weights) {
        int numActivationMaps = getNumActivationMaps(normalisedActivations);
        // Add dimensions to the weights for the multiplication
        weights = weights.reshape(numActivationMaps, 1, 1);

        return normalisedActivations.mul(weights);
    }

    private INDArray predictTargetClassWeights(INDArray maskedImages) {
        int numActivationMaps = getNumActivationMaps(maskedImages);
        log.info(String.format("Running prediction on %d masked images...", numActivationMaps));
//        INDArray targetClassWeights = Nd4j.zeros(numActivationMaps);
        double[] targetClassWeights = new double[numActivationMaps];

        ComputationGraph computationGraph = getComputationGraph();

        int totalIterations = numActivationMaps / batchSize;

        for (int iteration = 0; iteration < totalIterations; iteration++) {
            int fromIndex = iteration * batchSize;
            int toIndex = fromIndex + batchSize;
            INDArray maskedImageBatch = maskedImages.get(NDArrayIndex.interval(fromIndex, toIndex));
            // Run prediction
            INDArray output = computationGraph.outputSingle(maskedImageBatch);
            // Parse each prediction
            for (int miniBatchI = 0; miniBatchI < batchSize; miniBatchI++) {
                // Save the probability for the target class
                double classProbVal = output.getDouble(miniBatchI, targetClassID);
                targetClassWeights[fromIndex + miniBatchI] = classProbVal;
            }
        }
        return Nd4j.create(targetClassWeights);
    }

    private INDArray createMaskedImages(INDArray normalisedActivations, INDArray imageArr) {
        int numActivationMaps = getNumActivationMaps(normalisedActivations);

        // [1, 3, 224, 224] -> [3, 224, 224] - remove the minibatch dimension
        imageArr = Nd4j.squeeze(imageArr, 0);

        INDArray allMaskedImages = Nd4j.zeros(numActivationMaps, modelInputShape.getChannels(), modelInputShape.getHeight(), modelInputShape.getWidth());
        // Create the 512 masked images -
        // Multiply each normalized activation map with the image
        for (int i = 0; i < numActivationMaps; i++) {
            INDArray iActivationMap = normalisedActivations.get(NDArrayIndex.point(i));
            // [224, 224] -> [1, 224, 224] (is then broadcasted in the multiply method)
            iActivationMap = iActivationMap.reshape(1, modelInputShape.getHeight(), modelInputShape.getWidth());

            // [3, 224, 224] . [1, 224, 224] - actually create the masked image
            INDArray multiplied = imageArr.mul(iActivationMap);

            // Store the image
            INDArrayIndex[] index = new INDArrayIndex[] { NDArrayIndex.point(i) };
            allMaskedImages.put(index, multiplied);
        }

        return allMaskedImages;
    }

    private INDArray normalizeActivations(INDArray upsampledActivations) {
        // Normalize each of the 512 activation maps
        int numActivationMaps = getNumActivationMaps(upsampledActivations);

        for (int i = 0; i < numActivationMaps; i++) {
            INDArray tmpActivationMap = upsampledActivations.get(NDArrayIndex.point(i));
            double maxVal = tmpActivationMap.maxNumber().doubleValue();
            double minVal = tmpActivationMap.minNumber().doubleValue();
            double fudgeVal = 1e-5;
            double divisor = (maxVal - minVal) + fudgeVal;

            tmpActivationMap.divi(divisor);
        }
        return upsampledActivations;
    }

    private INDArray upsampleActivations(INDArray rawActivations) {
        // Create the new size array
        INDArray newSize = Nd4j.create(new int[] {(int) modelInputShape.getHeight(), (int) modelInputShape.getWidth()}, new long[] {2}, DataType.INT32);

        // Upsample the activations to match original image size
        NDImage ndImage = new NDImage();
        INDArray upsampledActivations = ndImage.imageResize(rawActivations, newSize, ImageResizeMethod.ResizeBicubic); // TODO try BiLinear

        // Drop the mini-batch size (1) from [1, 224, 224, 512]
        upsampledActivations = Nd4j.squeeze(upsampledActivations, 0);

        // Reshape back to [C, H, W] (easier to iterate over feature maps)
        return upsampledActivations.permute(2, 0, 1);
    }

    private INDArray getActivationsForImage(TransferLearningHelper transferLearningHelper, INDArray imageArr) {
        // Run the model on the image to get the activation maps
        DataSet imageDataset = new DataSet(imageArr, Nd4j.zeros(1));
        DataSet result = transferLearningHelper.featurize(imageDataset);
        INDArray rawActivations = result.getFeatures();

        // Must be channels last for the imageResize method
        return rawActivations.permute(0, 2, 3, 1);
    }

    private INDArray loadImage(File imageFile) {
        // Load the image
        NativeImageLoader loader = new NativeImageLoader(modelInputShape.getHeight(), modelInputShape.getWidth(), modelInputShape.getChannels());
        try {
            return loader.asMatrix(imageFile);
        } catch (IOException ex) {
            ex.printStackTrace();
            return null;
        }
    }

    public InputType.InputTypeConvolutional getModelInputShape() {
        return modelInputShape;
    }

    public void setModelInputShape(int[] shape) {
        setModelInputShape(Utils.decodeCNNShape(shape));
    }

    public void setModelInputShape(InputType.InputTypeConvolutional modelInputShape) {
        this.modelInputShape = modelInputShape;
    }

    /**
     * Assuming arr is in order [channels, height, width]
     * @param activationMaps
     * @return
     */
    private int getNumActivationMaps(INDArray activationMaps) {
        return (int) activationMaps.shape()[0];
    }
}
