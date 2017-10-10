package weka.dl4j.updater;

import weka.core.OptionMetadata;

/**
 * A WEKA version of DeepLearning4j's Sgd.
 *
 * @author Steven Lang
 * @version $Revision: 11711 $
 */
public class Sgd extends org.nd4j.linalg.learning.config.Sgd implements Updater {
    
    @OptionMetadata(
            displayName = "learningrate",
            description = "The learningrate to use (default = 0.001).",
            commandLineParamName = "learningRate", commandLineParamSynopsis = "-learningRate <double>",
            displayOrder = 12)
    @Override
    public double getLearningRate() {
        return super.getLearningRate();
    }
    
    @Override
    public void setLearningRate(double learningRate) {
        super.setLearningRate(learningRate);
    }
}
