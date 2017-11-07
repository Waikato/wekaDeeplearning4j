package weka.dl4j.updater;

import weka.core.OptionMetadata;
import weka.gui.ProgrammaticProperty;

/**
 * A WEKA version of DeepLearning4j's AdaGrad.
 *
 * @author Steven Lang
 */

public class AdaGrad extends org.nd4j.linalg.learning.config.AdaGrad implements Updater {
    private static final long serialVersionUID = 3881105990718165790L;

    @ProgrammaticProperty
    @Override
    public double getLearningRate() {
        return super.getLearningRate();
    }

    @ProgrammaticProperty
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
