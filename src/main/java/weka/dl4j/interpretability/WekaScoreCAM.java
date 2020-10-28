package weka.dl4j.interpretability;

import lombok.extern.log4j.Log4j2;
import weka.classifiers.functions.dl4j.Utils;
import weka.core.progress.ProgressManager;
import weka.dl4j.inference.PredictionClass;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.Buffer;

// TODO Document
@Log4j2
public class WekaScoreCAM extends AbstractCNNSaliencyMapWrapper {

    protected ScoreCAM scoreCAM;

    @Override
    public void processImage(File imageFile) {
        scoreCAM = new ScoreCAM();
        scoreCAM.setComputationGraph(getComputationGraph());
        scoreCAM.setBatchSize(batchSize);

        scoreCAM.setImageChannelsLast(zooModel.getChannelsLast()); // TODO check with non-zooModels
        scoreCAM.setModelInputShape(Utils.decodeCNNShape(zooModel.getShape()[0]));
        scoreCAM.setImagePreProcessingScaler(zooModel.getImagePreprocessingScaler());
        scoreCAM.setModelName(zooModel.getPrettyName());

        scoreCAM.addIterationsStartedListener(this::onIterationsStarted);
        scoreCAM.addIterationIncrementListener(this::onIterationIncremented);
        scoreCAM.addIterationsFinishedListeners(this::onIterationsFinished);

        scoreCAM.processImage(imageFile);
    }

    @Override
    public BufferedImage generateHeatmapToImage() {
        int[] targetClassIDs = getTargetClassIDsAsInt();
        var normalize = getNormalizeHeatmap();
        return scoreCAM.generateHeatmapToImage(targetClassIDs, normalize);
    }

    private PredictionClass[] getTestPredictionClasses(int[] targetClassIDs) {
        var result = new PredictionClass[targetClassIDs.length];

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
