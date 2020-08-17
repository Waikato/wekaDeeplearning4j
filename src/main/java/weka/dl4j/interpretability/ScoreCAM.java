package weka.dl4j.interpretability;

import lombok.extern.log4j.Log4j2;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.nd4j.enums.ImageResizeMethod;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.ops.NDImage;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

// TODO document
@Log4j2
public class ScoreCAM extends AbstractSaliencyMapGenerator {

    protected String activationMapLayer;

    public ScoreCAM(ComputationGraph model, String activationMapLayer) {
        super(model);
        this.activationMapLayer = activationMapLayer;
    }

    @Override
    public void generateForImage(File imageFile, int targetClassID) {
        TransferLearningHelper transferLearningHelper = new TransferLearningHelper(
                this.model.clone(), activationMapLayer);

        INDArray imageArr = loadImage(imageFile);

        INDArray rawActivations = getActivationsForImage(transferLearningHelper, imageArr);

        INDArray upsampledActivations = upsampleActivations(rawActivations);

        upsampledActivations = reshapeUpsampledActivations(upsampledActivations);

        INDArray normalisedActivations = normalizeActivations(upsampledActivations);

        INDArray maskedImages = createMaskedImages(normalisedActivations, imageArr);

        INDArray targetClassWeights = getTargetClassWeights(maskedImages, targetClassID);

        INDArray weightedActivationMaps = applyActivationMapWeights(normalisedActivations, targetClassWeights);

        INDArray postprocessedActivations = postprocessActivations(weightedActivationMaps);

        saveResults(imageArr, postprocessedActivations, imageFile.getName() + "_processed");
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
        return reluActivations.reshape(1, 224, 224);
    }

    private void postNormaliseMap(INDArray activationMap) {
        // Perform any normalizing of the activation maps
        // Scale the map to between 0 and 1 (so it can be multiplied on the image)
        double currMax = activationMap.maxNumber().doubleValue();
        double currMin = activationMap.minNumber().doubleValue();
        System.out.println(String.format("Prev max: %.4f, prev min: %.4f", currMax, currMin));

//        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0.0, 1.0);
//        normalizer.fit(new DataSet(activationMap, Nd4j.zeros(1)));
//        normalizer.transform(activationMap);

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

    private INDArray getTargetClassWeights(INDArray maskedImages, int targetClassID) {
        int numActivationMaps = getNumActivationMaps(maskedImages);

        INDArray targetClassWeights = Nd4j.zeros(numActivationMaps);

        for (int i = 0; i < numActivationMaps; i++) {
            INDArray maskedImage = maskedImages.get(NDArrayIndex.point(i));
            // Needs the minibatch added back in for prediction
            maskedImage = maskedImage.reshape(1, 3, 224, 224);

            // Run prediction
            INDArray output = model.outputSingle(maskedImage);
            // Save the probability for the target class
            double classProbVal = output.getDouble(targetClassID);
            targetClassWeights.putScalar(i, classProbVal);
        }
        return targetClassWeights;
    }

    private INDArray createMaskedImages(INDArray normalisedActivations, INDArray imageArr) {
        int numActivationMaps = getNumActivationMaps(normalisedActivations);

        // [1, 3, 224, 224] -> [3, 224, 224] - remove the minibatch dimension
        imageArr = Nd4j.squeeze(imageArr, 0);

        INDArray allMaskedImages = Nd4j.zeros(numActivationMaps, 3, 224, 224);
        // Create the 512 masked images -
        // Multiply each normalized activation map with the image
        for (int i = 0; i < numActivationMaps; i++) {
            INDArray iActivationMap = normalisedActivations.get(NDArrayIndex.point(i));
            // [224, 224] -> [1, 224, 224] (is then broadcasted in the multiply method)
            iActivationMap = iActivationMap.reshape(1, 224, 224);

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
        INDArray newSize = Nd4j.create(new int[] {224, 224}, new long[] {2}, DataType.INT32);

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
        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        try {
            return loader.asMatrix(imageFile);
        } catch (IOException ex) {
            ex.printStackTrace();
            return null;
        }
    }

    private int getNumActivationMaps(INDArray activationMaps) {
        return (int) activationMaps.shape()[0];
    }
}
