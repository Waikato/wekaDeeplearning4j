package weka.dl4j.interpretability;

import lombok.extern.log4j.Log4j2;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.progress.ProgressManager;
import weka.dl4j.inference.PredictionClass;

import java.awt.image.BufferedImage;
import java.io.File;

// TODO Document
@Log4j2
public class WekaScoreCAM extends AbstractCNNSaliencyMapWrapper {

    protected ScoreCAM scoreCAM;

    @Override
    public void processImage(File imageFile) {
        scoreCAM = new ScoreCAM();
        scoreCAM.setBatchSize(batchSize);
        Dl4jMlpClassifier classifier = getDl4jMlpClassifier();
        scoreCAM.setComputationGraph(classifier.getModel());
        scoreCAM.setModelInputShape(classifier.getInputShape(getCustomModelSetup()));
        scoreCAM.setModelName(classifier.getModelName());

        if (classifier.useZooModel()) {
            scoreCAM.setImageChannelsLast(classifier.getZooModel().getChannelsLast());
            scoreCAM.setImagePreProcessingScaler(classifier.getZooModel().getImagePreprocessingScaler());
        }

        scoreCAM.addIterationsStartedListener(this::onIterationsStarted);
        scoreCAM.addIterationIncrementListener(this::onIterationIncremented);
        scoreCAM.addIterationsFinishedListeners(this::onIterationsFinished);

        scoreCAM.processImage(imageFile);
    }

    @Override
    public BufferedImage generateHeatmapToImage() {
        int[] targetClassIDs = getTargetClassIDsAsInt();
        boolean normalize = getNormalizeHeatmap();
        return scoreCAM.generateHeatmapToImage(targetClassIDs, getClassMap(), normalize);
    }

    private PredictionClass[] getTestPredictionClasses(int[] targetClassIDs) {
        PredictionClass[] result = new PredictionClass[targetClassIDs.length];

        for (int i = 0; i < targetClassIDs.length; i++)
        {
            result[i] = new PredictionClass(
                    targetClassIDs[i],
                    String.format("Test class %d", i)
            );
        }
        return result;
    }

    private void onIterationsStarted(int maxIterations) {
        progressManager = new ProgressManager(maxIterations, String.format("Calculating Saliency Map with a batch size of %d...", scoreCAM.getBatchSize()));
        progressManager.start();
    }

    private void onIterationIncremented() {
        progressManager.increment();
    }

    private void onIterationsFinished() {
        progressManager.finish();
    }

}
