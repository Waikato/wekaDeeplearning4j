package weka.dl4j.updater;

import weka.core.OptionMetadata;


/**
 * A WEKA version of DeepLearning4j's Adam.
 *
 * @author Steven Lang
 * @version $Revision: 11711 $
 */
public class Adam extends org.nd4j.linalg.learning.config.Adam implements Updater {
    @OptionMetadata(
            displayName = "learningrate",
            description = "The learningrate to use (default = " + DEFAULT_ADAM_EPSILON + ").",
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
            displayName = "beta1MeanDecay",
            description = "The mean decay (default = " + DEFAULT_ADAM_BETA1_MEAN_DECAY + ").",
            commandLineParamName = "beta1MeanDecay", commandLineParamSynopsis = "-beta1MeanDecay <double>",
            displayOrder = 1)
    @Override
    public double getBeta1() {
        return super.getBeta1();
    }
    
    @Override
    public void setBeta1(double beta1) {
        super.setBeta1(beta1);
    }
    
    @OptionMetadata(
            displayName = "beta2VarDecay",
            description = "The var decay (default = " + DEFAULT_ADAM_BETA1_MEAN_DECAY + ").",
            commandLineParamName = "beta2VarDecay", commandLineParamSynopsis = "-beta2VarDecay <double>",
            displayOrder = 2)
    @Override
    public double getBeta2() {
        return super.getBeta2();
    }
    
    @Override
    public void setBeta2(double beta2) {
        super.setBeta2(beta2);
    }
    
    @OptionMetadata(
            displayName = "epsilon",
            description = "The epsilon parameter (default = " + DEFAULT_ADAM_BETA1_MEAN_DECAY + ").",
            commandLineParamName = "epsilon", commandLineParamSynopsis = "-epsilon <double>",
            displayOrder = 3)
    @Override
    public double getEpsilon() {
        return super.getEpsilon();
    }
    
    @Override
    public void setEpsilon(double epsilon) {
        super.setEpsilon(epsilon);
    }
}
