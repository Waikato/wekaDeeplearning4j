package weka.dl4j.updater;

import weka.core.OptionMetadata;
import weka.gui.ProgrammaticProperty;


/**
 * A WEKA version of DeepLearning4j's RmsProp.
 *
 * @author Steven Lang
 */
public class RmsProp extends org.nd4j.linalg.learning.config.RmsProp implements Updater {
    private static final long serialVersionUID = 7400615175279701837L;
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
            displayName = "rmsDecay",
            description = "The rms decay (default = " + DEFAULT_RMSPROP_RMSDECAY + ").",
            commandLineParamName = "rmsDecay", commandLineParamSynopsis = "-rmsDecay <double>",
            displayOrder = 1)
    @Override
    public double getRmsDecay() {
        return super.getRmsDecay();
    }

    @Override
    public void setRmsDecay(double rmsDecay) {
        super.setRmsDecay(rmsDecay);
    }

    @OptionMetadata(
            displayName = "epsilon",
            description = "The epsilon parameter (default = " + DEFAULT_RMSPROP_EPSILON + ").",
            commandLineParamName = "epsilon", commandLineParamSynopsis = "-epsilon <double>",
            displayOrder = 2)
    @Override
    public double getEpsilon() {
        return super.getEpsilon();
    }

    @Override
    public void setEpsilon(double epsilon) {
        super.setEpsilon(epsilon);
    }
}
