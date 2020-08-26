package weka.dl4j.interpretability;

import weka.classifiers.functions.dl4j.Utils;
import weka.core.progress.ProgressManager;

import java.io.File;

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
        progressManager.start();
    }

    private void onIterationIncremented() {
        progressManager.increment();
    }

    private void onIterationsFinished() {
        progressManager.finish();
    }

}
