package weka.dl4j.updater;

import weka.core.OptionMetadata;

/**
 * A WEKA version of DeepLearning4j's AdaGrad.
 *
 * @author Steven Lang
 * @version $Revision: 11711 $
 */

public class AdaGrad extends org.nd4j.linalg.learning.config.AdaGrad implements Updater {
    @OptionMetadata(
            displayName = "learningrate",
            description = "The learningrate to use (default = " + DEFAULT_ADAGRAD_LEARNING_RATE + ").",
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
            displayName = "epsilon",
            description = "The epsilon parameter (default = " + DEFAULT_ADAGRAD_EPSILON + ").",
            commandLineParamName = "epsilon", commandLineParamSynopsis = "-epsilon <double>",
            displayOrder = 1)
    @Override
    public double getEpsilon() {
        return super.getEpsilon();
    }
    
    @Override
    public void setEpsilon(double epsilon) {
        super.setEpsilon(epsilon);
    }
}
