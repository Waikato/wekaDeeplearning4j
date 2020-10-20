package weka.dl4j.interpretability;

import lombok.extern.log4j.Log4j2;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.inputs.InputType;
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
import weka.dl4j.interpretability.listeners.IterationIncrementListener;
import weka.dl4j.interpretability.listeners.IterationsStartedListener;
import weka.dl4j.interpretability.listeners.IterationsFinishedListener;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

// TODO document
@Log4j2
public class ScoreCAM extends AbstractCNNSaliencyMapGenerator {

    protected int outsideBorder = 25;

    protected int insidePadding = 20;

    protected int fontSpacing = 12;

    protected INDArray originalImageArr, preprocessedImageArr, softmaxOnMaskedImages, normalisedActivations;

    public void processMaskedImages(File imageFile) {
        originalImageArr = loadImage(imageFile);
        // Preprocess the image if the model requires it
        preprocessedImageArr = preprocessImage(originalImageArr);
        // Get the original set of activation maps by taking the activations
        // from the final convolution layer when running the image through the model
        INDArray rawActivations = getActivationsForImage(preprocessedImageArr);

        // Upsample the activations to match the original image size
        INDArray upsampledActivations = upsampleActivations(rawActivations);

        // Normalise them between 0 and 1 (so they can be multiplied with the images)
        normalisedActivations = normalizeActivationMaps(upsampledActivations);

        // Create the set of masked images by multiplying each (upsampled, normalized) activation map with the original image
        INDArray maskedImages = createMaskedImages(normalisedActivations, preprocessedImageArr);

        // Get the softmax score for the target class ID when running the masked images through the model
        softmaxOnMaskedImages = predictOnMaskedImages(maskedImages);
    }

    @Override
    public void processImage(File imageFile) {
        processMaskedImages(imageFile);
    }

    @Override
    public void generateOutputMap() {
        allImages.clear();
        for (int targetClassID : getTargetClassIDs()) {
            int actualClassID = calculateTargetClassID(preprocessedImageArr, targetClassID);

            INDArray targetClassWeights = calculateTargetClassWeights(softmaxOnMaskedImages, actualClassID);

            // Weight each activation map using the previously acquired softmax scores
            INDArray weightedActivationMaps = applyActivationMapWeights(normalisedActivations, targetClassWeights);

            // Sum the activation maps into one map and normalise the saliency map values to between [0, 1]
            INDArray postprocessedActivations = postprocessActivations(weightedActivationMaps);

            createFinalImages(originalImageArr, postprocessedActivations);
        }
    }

    private INDArray preprocessImage(INDArray imageArr) {
        ImagePreProcessingScaler scaler = getImagePreProcessingScaler();

        if (scaler == null) {
            log.debug("No image preprocessing required");
            return imageArr;
        }

        INDArray preprocessed = imageArr.dup();

        log.debug("Applying image preprocessing...");
        scaler.transform(preprocessed);
        return preprocessed;
    }

    private int calculateTargetClassID(INDArray imageArr, int targetClassID) {
        if (targetClassID != -1) {
            // Target class has already been set, don't calculate it
            return targetClassID;
        }
        // Otherwise run the model on the image, and choose argmax as the target class
        INDArray output = modelOutputSingle(imageArr);
        int argMax = output.argMax(1).getNumber(0).intValue();
        return argMax;
    }

    private INDArray calculateTargetClassWeights(INDArray softmaxOutput, int targetClassID) {
        return softmaxOutput.getColumn(targetClassID).dup();
    }

    private boolean isNDArrayChannelsLast(INDArray in) {
        long[] shape = in.shape();
        long minibatch = shape[0];
        long val1 = shape[1];
        long val2 = shape[2];
        long val3 = shape[3];

        // Returns true for in = [1, 224, 224, 3] (224 == 224 and 224 != 3)
        return val1 == val2 && val2 != val3;
    }

    private INDArray permuteToSuit(INDArray in, boolean shouldBeChannelsLast) {
        // Check if the array we're being passed is is channels last
        boolean inChannelsLast = isNDArrayChannelsLast(in);

        // If 'in' is channels last and it should be channels last then leave it
        // (and similarly for if it's channels first and we're expecting channels first
        if (inChannelsLast == shouldBeChannelsLast) {
            log.debug("No permutation necessary");
            return in;
        }

        INDArray permuted = in.permute(0,3,2,1);
        log.debug(String.format("Permuted NDArray from %s to %s",
                Arrays.toString(in.shape()), Arrays.toString(permuted.shape())));
        return permuted.dup();
    }

    private INDArray preModelCheck(INDArray in) {
        boolean modelExpectsChannelsLast = isImageChannelsLast();
        return permuteToSuit(in, modelExpectsChannelsLast);
    }

    private INDArray postModelCheck(INDArray in) {
        return permuteToSuit(in, false);
    }

    private INDArray modelOutputSingle(INDArray in) {
        // Transform it before we pass to the model if necessary
        // Default should be [NCHW] but may need to permute to [NHWC]
        INDArray preChecked = preModelCheck(in);
        INDArray results = getComputationGraph().outputSingle(preChecked);
        return results;
    }

    /**
     * Gets the activation map layer to use as the image masks - this is taken as the final convolution layer
     * @return
     */
    private String getActivationMapLayer() {
        Layer[] layers = getComputationGraph().getLayers();
        Layer[] convLayers = Arrays.stream(layers).filter(x -> x instanceof ConvolutionLayer).toArray(Layer[]::new);
        int count = convLayers.length;
        String layerName = convLayers[count - 1].getConfig().getLayerName();
        log.debug("Auto selected activation layer name is " + layerName);
        return layerName;
    }

    private void createHeatmap(INDArray postprocessedActivations) {
        Color[] gradientColors = Gradient.GRADIENT_PLASMA;
        heatmap = new BufferedImage(
                (int) modelInputShape.getWidth(),
                (int) modelInputShape.getHeight(),
                BufferedImage.TYPE_INT_ARGB);

        Graphics2D g = heatmap.createGraphics();

        for (int row = 0; row < modelInputShape.getHeight(); row++) {
            for (int col = 0; col < modelInputShape.getWidth(); col++) {
                double normVal = postprocessedActivations.getDouble(row, col);
                int colorIndex = (int) Math.floor(normVal * (gradientColors.length - 1));

                // Limit it on the highest color
                // - if we're not normalizing the heatmap then this can cause IndexOutOfBoundsException otherwise
                // E.g., colorIndex = 610 with 500 gradientColors - colorIndex is now 499.
                colorIndex = Math.min(colorIndex, gradientColors.length - 1);

                Color color = gradientColors[colorIndex];
                g.setColor(color);
                g.fillRect(col, row, 1, 1);
            }
        }
        g.dispose();
    }

    private void createOriginalImage(INDArray imageArr) {
        originalImage = imageFromINDArray(imageArr);
    }

    private void createFinalImages(INDArray imageArr, INDArray postprocessedActivations) {
        createHeatmap(postprocessedActivations);
        createOriginalImage(imageArr);
        createOverlaidHeatmap();
        createCompositeImage();
        allImages.add(getCompositeImage());
    }

    private int calculateCompositeWidth() {
        // Outside margins plus images plus padding
        return outsideBorder * 2 + (insidePadding * 2) + ((int) modelInputShape.getWidth() * 3);
    }

    private int calculateCompositeHeight() {
        // Outside margins plus image height plus space for text
        return outsideBorder * 2 + (int) modelInputShape.getHeight() + (fontSpacing * 2);
    }

    private void createCompositeImage() {
        int width = calculateCompositeWidth();
        int height = calculateCompositeHeight();

        compositeImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = compositeImage.createGraphics();

        g.setColor(Color.WHITE);
        g.fillRect(0, 0, width, height);
        g.setComposite(AlphaComposite.SrcOver);

        int leftX = outsideBorder;
        int leftY = outsideBorder;
        // Draw the left image
        g.drawImage(getOriginalImage(), leftX, leftY, null);

        // Draw the center image
        int midX = (int) (outsideBorder + modelInputShape.getWidth() + insidePadding);
        int midY = outsideBorder;
        g.drawImage(getHeatmap(), midX, midY, null);

        // Draw the right image
        int rightX = (int) (outsideBorder + (modelInputShape.getWidth() * 2) + (insidePadding * 2));
        int rightY = outsideBorder;
        g.drawImage(getHeatmapOnImage(), rightX, rightY, null);

        // Draw the map info
        int textX = leftX;
        int textY = leftY + (int) modelInputShape.getHeight() + (fontSpacing * 2);
        g.setColor(Color.BLACK);
//        g.setFont(new Font("Serif", Font.PLAIN, fontSpacing));
        g.drawString(String.format("Image file: %s       Saliency Map Method: ScoreCAM       Base model: %s",
                getInputFilename(), getModelName()), textX, textY);

        g.dispose();
    }

    private void createOverlaidHeatmap() {
        // From https://www.reddit.com/r/javahelp/comments/2ufc0m/how_do_i_overlay_2_bufferedimages_and_set_the/co7yrv9?utm_source=share&utm_medium=web2x&context=3
        heatmapOnImage = new BufferedImage(224, 224, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = heatmapOnImage.createGraphics();

        // Clear the image (optional)
        g.setComposite(AlphaComposite.Clear);
        g.fillRect(0,0, 224, 224);

        // Draw the background image
        g.setComposite(AlphaComposite.SrcOver);
        g.drawImage(originalImage, 0, 0, null);

        // Draw the overlay image
        float alpha = 0.7f;
        g.setComposite(AlphaComposite.SrcOver.derive(alpha));
        g.drawImage(heatmap, 0, 0, null);

        g.dispose();
    }

    private INDArray postprocessActivations(INDArray weightedActivationMaps) {
        // Sum all maps to get one 224x224 map - [numActivationMaps, 224, 224] -> [224, 224]
        INDArray summed = weightedActivationMaps.sum(0);
        // Perform pixel-wise RELU
        INDArray reluActivations = Transforms.relu(summed);

        if (getNormalizeHeatmap())
            normalize2x2ArrayI(reluActivations);

        return reluActivations;
    }

    private INDArray normalizeActivationMaps(INDArray upsampledActivations) {
        // Normalize each of the 512 activation maps
        int numActivationMaps = getNumActivationMaps(upsampledActivations);

        // Perform any normalizing of the activation maps
        // Scale the map to between 0 and 1 (so it can be multiplied on the image)
        double currMax = upsampledActivations.maxNumber().doubleValue();
        double currMin = upsampledActivations.minNumber().doubleValue();
        log.debug(String.format("All activation maps: prev max: %.4f, prev min: %.4f", currMax, currMin));

        for (int i = 0; i < numActivationMaps; i++) {
            INDArray tmpActivationMap = upsampledActivations.get(NDArrayIndex.point(i));
            normalize2x2ArrayI(tmpActivationMap);
        }

        double newMax = upsampledActivations.maxNumber().doubleValue();
        double newMin = upsampledActivations.minNumber().doubleValue();
        log.debug(String.format("All activation maps: new max: %.4f, new min: %.4f", newMax, newMin));

        return upsampledActivations;
    }

    private void normalize2x2ArrayI(INDArray array) {
        double currMax = array.maxNumber().doubleValue();
        double currMin = array.minNumber().doubleValue();

        array.subi(currMin);
        array.divi(currMax - currMin);
    }

    private INDArray applyActivationMapWeights(INDArray normalisedActivations, INDArray weights) {
        int numActivationMaps = getNumActivationMaps(normalisedActivations);
        // Add dimensions to the weights for the multiplication
        weights = weights.reshape(numActivationMaps, 1, 1);

        return normalisedActivations.mul(weights);
    }

    private int getNumIterations(int numActivationMaps) {
        if (numActivationMaps % batchSize == 0) {
            return numActivationMaps / batchSize;
        } else {
            return (numActivationMaps / batchSize) + 1;
        }
    }

    private int getSafeToIndex(int fromIndex, int numActivationMaps) {
        int toIndex = fromIndex + batchSize;
        if (toIndex >= numActivationMaps) {
            toIndex = numActivationMaps;
        }
        return toIndex;
    }

    private void broadcastIterationsStarted(int totalIterations) {
        for (IterationsStartedListener listener : iterationsStartedListeners) {
            listener.iterationsStarted(totalIterations);
        }
    }

    private void broadcastIterationIncremented() {
        for (IterationIncrementListener listener : iterationIncrementListeners) {
            listener.iterationIncremented();
        }
    }

    private void broadcastIterationsFinished() {
        for (IterationsFinishedListener listener : iterationsFinishedListeners) {
            listener.iterationsFinished();
        }
    }

    private INDArray predictOnMaskedImages(INDArray maskedImages) {
        int numActivationMaps = getNumActivationMaps(maskedImages);
        log.info(String.format("Running prediction on %d masked images with a batch size of %d", numActivationMaps, batchSize));
        INDArray softmaxOnMaskedImages = null;

        int totalIterations = getNumIterations(numActivationMaps);

        // Fire the iterations started listeners
        broadcastIterationsStarted(totalIterations);

        for (int iteration = 0; iteration < totalIterations; iteration++) {
            int fromIndex = iteration * batchSize;
            int toIndex = getSafeToIndex(fromIndex, numActivationMaps);
            // In the case of a partial batch (e.g., the final batch using a batch size of 10 with 2048 activation maps,
            // we need to know what the *actual* batch size of this iteration is (8, in this example)
            int actualBatchSize = toIndex - fromIndex;
            INDArray maskedImageBatch = maskedImages.get(NDArrayIndex.interval(fromIndex, toIndex));
            // Run prediction
            INDArray output = modelOutputSingle(maskedImageBatch);

            if (softmaxOnMaskedImages == null) {
                int numClasses = (int) output.shape()[1];
                softmaxOnMaskedImages = Nd4j.zeros(numActivationMaps, numClasses);
            }

            // Parse each prediction
            for (int miniBatchI = 0; miniBatchI < actualBatchSize; miniBatchI++) {
                // Save the probability for the target class
                INDArray row = output.getRow(miniBatchI);
                softmaxOnMaskedImages.putRow(fromIndex + miniBatchI, row);
            }
            // Fire the iteration increment listeners
            broadcastIterationIncremented();
        }
        // Finish the iterations
        broadcastIterationsFinished();
        return softmaxOnMaskedImages;
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

    private INDArray upsampleActivations(INDArray rawActivations) {
        // Create the new size array
        INDArray newSize = Nd4j.create(new int[] {(int) modelInputShape.getHeight(), (int) modelInputShape.getWidth()}, new long[] {2}, DataType.INT32);

        // Upsample the activations to match original image size
        NDImage ndImage = new NDImage();
        // Should use Bilinear interpolation but that seems to smooth over activations, removing them entirely
        INDArray upsampledActivations = ndImage.imageResize(rawActivations, newSize, ImageResizeMethod.ResizeBicubic);

        // Drop the mini-batch size (1) from [1, 224, 224, 512]
        upsampledActivations = Nd4j.squeeze(upsampledActivations, 0);

        // Reshape back to [C, H, W] (easier to iterate over feature maps)
        return upsampledActivations.permute(2, 0, 1);
    }

    private INDArray getActivationsForImage(INDArray imageArr) {
        // Set up the model and image
        String activationMapLayer = getActivationMapLayer();

        TransferLearningHelper transferLearningHelper = new TransferLearningHelper(
                getComputationGraph().clone(), activationMapLayer);

        imageArr = preModelCheck(imageArr);
        // Run the model on the image to get the activation maps
        DataSet imageDataset = new DataSet(imageArr, Nd4j.zeros(1));
        DataSet result = transferLearningHelper.featurize(imageDataset);
        INDArray rawActivations = result.getFeatures();
        rawActivations = postModelCheck(rawActivations);

        log.debug("Raw activation shape is " + Arrays.toString(rawActivations.shape()));

        // Must be channels last for the imageResize method
        return rawActivations.permute(0, 2, 3, 1);
    }

    private INDArray loadImage(File imageFile) {
        setInputFilename(imageFile.getName());
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

    /**
     * Takes an INDArray containing an image loaded using the native image loader
     * libraries associated with DL4J, and converts it into a BufferedImage.
     * The INDArray contains the color values split up across three channels (RGB)
     * and in the integer range 0-255.
     *
     * @param array INDArray containing an image in order [N, C, H, W] or [C, H, W]
     * @return BufferedImage
     */
    public static BufferedImage imageFromINDArray(INDArray array) {
        long[] shape = array.shape();

        boolean is4d = false;
        String dimString = "3D";

        if (shape.length == 4) {
            is4d = true;
            dimString = "4D";
        }

        log.debug(String.format("Converting %s INDArray to image...", dimString));

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
}
