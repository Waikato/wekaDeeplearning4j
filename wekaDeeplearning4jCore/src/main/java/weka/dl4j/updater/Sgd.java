package weka.dl4j.updater;

import weka.core.OptionMetadata;

/**
 * A WEKA version of DeepLearning4j's Sgd.
 *
 * @author Steven Lang
 */
public class Sgd extends org.nd4j.linalg.learning.config.Sgd implements Updater {

    private static final long serialVersionUID = 1852959048173443658L;

    @OptionMetadata(
            displayName = "learningrate",
            description = "The learningrate to use (default = " + DEFAULT_SGD_LR + ").",
            commandLineParamName = "learningRate", commandLineParamSynopsis = "-learningRate <double>",
            displayOrder = 0)
    @Override
    public double getLearningRate() {
        return super.getLearningRate();
    }
    
    @Override
    public void setLearningRate(double learningRate) {
        super.setLearningRate(learningRate);
    }
}
