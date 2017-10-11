package weka.dl4j.updater;


import weka.core.OptionMetadata;

/**
 * A WEKA version of DeepLearning4j's AdaDelta.
 *
 * @author Steven Lang
 * @version $Revision: 11711 $
 */
public class AdaDelta extends org.nd4j.linalg.learning.config.AdaDelta implements Updater {
    @OptionMetadata(
            displayName = "rho",
            description = "The rho parameter (default = " + DEFAULT_ADADELTA_RHO + ").",
            commandLineParamName = "rho", commandLineParamSynopsis = "-rho <double>",
            displayOrder = 0)
    @Override
    public double getRho() {
        return super.getRho();
    }
    
    @Override
    public void setRho(double rho) {
        super.setRho(rho);
    }
    
    
    @OptionMetadata(
            displayName = "epsilon",
            description = "The epsilon parameter (default = " + DEFAULT_ADADELTA_EPSILON + ").",
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
