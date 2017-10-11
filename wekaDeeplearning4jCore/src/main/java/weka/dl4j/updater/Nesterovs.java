package weka.dl4j.updater;


import weka.core.OptionMetadata;

/**
 * A WEKA version of DeepLearning4j's Nesterovs.
 *
 * @author Steven Lang
 * @version $Revision: 11711 $
 */
public class Nesterovs extends org.nd4j.linalg.learning.config.Nesterovs implements Updater {
    @OptionMetadata(
            displayName = "learningrate",
            description = "The learningrate to use (default = " + DEFAULT_NESTEROV_LEARNING_RATE + ").",
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
    
    @OptionMetadata(
            displayName = "momentum",
            description = "The momentum (default = " + DEFAULT_NESTEROV_MOMENTUM + ").",
            commandLineParamName = "momentum", commandLineParamSynopsis = "-momentum <double>",
            displayOrder = 1)
    @Override
    public double getMomentum() {
        return super.getMomentum();
    }
    
    @Override
    public void setMomentum(double momentum) {
        super.setMomentum(momentum);
    }
}
