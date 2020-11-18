package weka.dl4j.interpretability;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import weka.dl4j.inference.Prediction;
import weka.dl4j.inference.PredictionClass;
import weka.dl4j.interpretability.listeners.IterationIncrementListener;
import weka.dl4j.interpretability.listeners.IterationsFinishedListener;
import weka.dl4j.interpretability.listeners.IterationsStartedListener;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Generic class for a saliency map generator.
 * Only implemented by ScoreCAM currently so some fields may need to be refactored when implementing a new
 * generator method.
 */
public abstract class AbstractCNNSaliencyMapGenerator {

    protected ComputationGraph computationGraph = null;

    protected int batchSize = 1;

    protected ImagePreProcessingScaler imagePreProcessingScaler = null;

    protected boolean imageChannelsLast = false;

    protected InputType.InputTypeConvolutional modelInputShape;

    protected String modelName;

    protected String inputFilename;

    protected int outsideMargin = 25;

    protected int insidePadding = 20;

    protected int fontSpacing = 12;

    public abstract void processImage(File imageFile);

    public abstract BufferedImage generateHeatmapToImage(int[] targetClasses, String[] classMap, boolean normalize);

    /**
     * Event listeners
     */
    protected List<IterationsStartedListener> iterationsStartedListeners = new ArrayList<>();

    protected List<IterationIncrementListener> iterationIncrementListeners = new ArrayList<>();

    protected List<IterationsFinishedListener> iterationsFinishedListeners = new ArrayList<>();

    public void addIterationsStartedListener(IterationsStartedListener listener) {
        iterationsStartedListeners.add(listener);
    }

    public void addIterationIncrementListener(IterationIncrementListener listener) {
        iterationIncrementListeners.add(listener);
    }

    public void addIterationsFinishedListeners(IterationsFinishedListener listener) {
        iterationsFinishedListeners.add(listener);
    }

    /**
     * Getters and setters
     */
    public ComputationGraph getComputationGraph() {
        return computationGraph;
    }

    public void setComputationGraph(ComputationGraph computationGraph) {
        this.computationGraph = computationGraph;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public ImagePreProcessingScaler getImagePreProcessingScaler() {
        return imagePreProcessingScaler;
    }

    public void setImagePreProcessingScaler(ImagePreProcessingScaler imagePreProcessingScaler) {
        this.imagePreProcessingScaler = imagePreProcessingScaler;
    }

    public boolean isImageChannelsLast() {
        return imageChannelsLast;
    }

    public void setImageChannelsLast(boolean imageChannelsLast) {
        this.imageChannelsLast = imageChannelsLast;
    }

    public InputType.InputTypeConvolutional getModelInputShape() {
        return modelInputShape;
    }

    public void setModelInputShape(InputType.InputTypeConvolutional modelInputShape) {
        this.modelInputShape = modelInputShape;
    }

    public String getModelName() {
        return modelName;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public String getInputFilename() {
        return inputFilename;
    }

    public void setInputFilename(String inputFilename) {
        this.inputFilename = inputFilename;
    }
}
