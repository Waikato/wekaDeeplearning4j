package weka.dl4j.updater;


import weka.core.OptionMetadata;
import weka.gui.ProgrammaticProperty;

/**
 * A WEKA version of DeepLearning4j's Nesterovs.
 *
 * @author Steven Lang
 */
public class Nesterovs extends org.nd4j.linalg.learning.config.Nesterovs implements Updater {
    private static final long serialVersionUID = 927121528229628203L;
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
