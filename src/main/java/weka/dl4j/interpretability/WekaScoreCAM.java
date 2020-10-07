package weka.dl4j.interpretability;

import lombok.extern.log4j.Log4j2;
import weka.classifiers.functions.dl4j.Utils;
import weka.core.progress.ProgressManager;

import javax.imageio.ImageIO;
import java.awt.*;
import java.io.File;

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

        scoreCAM.processMaskedImages(imageFile);
    }

    private void generate() {
        scoreCAM.setTargetClassID(getTargetClassID());
        scoreCAM.setNormalizeHeatmap(getNormalizeHeatmap());
        scoreCAM.generateOutputMap();
    }

    @Override
    public void generateOutputMap() {
        generate();
        saveResult();
    }

    @Override
    public Image generateOutputMapToImage() {
        generate();
        return scoreCAM.getCompositeImage();
    }

    private void saveResult() {
        if (Utils.notDefaultFileLocation(getOutputFile())) {
            log.info(String.format("Output file location = %s", getOutputFile()));
            try {
                ImageIO.write(scoreCAM.getCompositeImage(), "png", getOutputFile());
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        } else {
            log.error("No output file location given - not saving saliency map");
        }
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
