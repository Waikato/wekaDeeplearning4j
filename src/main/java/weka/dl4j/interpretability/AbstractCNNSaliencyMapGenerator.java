package weka.dl4j.interpretability;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import weka.dl4j.interpretability.listeners.IterationIncrementListener;
import weka.dl4j.interpretability.listeners.IterationsFinishedListener;
import weka.dl4j.interpretability.listeners.IterationsStartedListener;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

// TODO document
public abstract class AbstractCNNSaliencyMapGenerator {

    protected ComputationGraph computationGraph = null;

    protected int targetClassID = -1;

    protected int batchSize = 1;

    protected ImagePreProcessingScaler imagePreProcessingScaler = null;

    protected boolean imageChannelsLast = false;

    protected InputType.InputTypeConvolutional modelInputShape;

    protected BufferedImage originalImage;

    protected BufferedImage heatmap;

    protected BufferedImage heatmapOnImage;

    protected BufferedImage compositeImage;

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

    public abstract void generateForImage(File imageFile);

    public ComputationGraph getComputationGraph() {
        return computationGraph;
    }

    public void setComputationGraph(ComputationGraph computationGraph) {
        this.computationGraph = computationGraph;
    }

    public int getTargetClassID() {
        return targetClassID;
    }

    public void setTargetClassID(int targetClassID) {
        this.targetClassID = targetClassID;
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

    public BufferedImage getOriginalImage() {
        return originalImage;
    }

    public BufferedImage getHeatmap() {
        return heatmap;
    }

    public BufferedImage getHeatmapOnImage() {
        return heatmapOnImage;
    }

    public BufferedImage getCompositeImage() {
        return compositeImage;
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
}
