package weka.dl4j.interpretability;

import weka.classifiers.functions.dl4j.Utils;
import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.progress.ProgressManager;
import weka.dl4j.interpretability.listeners.IterationIncrementListener;
import weka.dl4j.interpretability.listeners.IterationsFinishedListener;
import weka.dl4j.interpretability.listeners.IterationsStartedListener;
import weka.dl4j.zoo.AbstractZooModel;

import java.io.File;
import java.io.Serializable;
import java.util.Enumeration;

// TODO Document
public class WekaScoreCAM extends AbstractSaliencyMapWrapper {

    /**
     * Displays progress of the current process (feature extraction, training, etc.)
     */
    protected ProgressManager progressManager;

    @Override
    public void run(File imageFile) {
        ScoreCAM scoreCAM = new ScoreCAM();
        scoreCAM.setComputationGraph(getComputationGraph());
        scoreCAM.setBatchSize(batchSize);
        scoreCAM.setTargetClassID(targetClassID);

        scoreCAM.setImageChannelsLast(zooModel.getChannelsLast());
        scoreCAM.setModelInputShape(Utils.decodeCNNShape(zooModel.getShape()[0]));
        scoreCAM.setImagePreProcessingScaler(zooModel.getImagePreprocessingScaler());

        scoreCAM.addIterationsStartedListener(this::onIterationsStarted);
        scoreCAM.addIterationIncrementListener(this::onIterationIncremented);
        scoreCAM.addIterationsFinishedListeners(this::onIterationsFinished);

        scoreCAM.generateForImage(imageFile);
    }

    private void onIterationsStarted(int maxIterations) {
        progressManager = new ProgressManager(maxIterations, "Calculating Saliency Map...");
        progressManager.show();
    }

    private void onIterationIncremented() {
        progressManager.increment();
    }

    private void onIterationsFinished() {
        progressManager.finish();
    }

}
