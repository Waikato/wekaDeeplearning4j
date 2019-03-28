
package weka.dl4j.updater;

import java.util.Enumeration;
import lombok.extern.log4j.Log4j2;
import weka.core.Option;
import weka.core.OptionMetadata;
import weka.gui.ProgrammaticProperty;

/**
 * A WEKA version of DeepLearning4j's AdaDelta.
 *
 * @author Steven Lang
 */
@Log4j2
public class AdaDelta extends Updater< org.nd4j.linalg.learning.config.AdaDelta > {
  private static final long serialVersionUID = -5776515704843860182L;

  @OptionMetadata(
    displayName = "rho",
    description = "The rho parameter (default = " + org.nd4j.linalg.learning.config.AdaDelta.DEFAULT_ADADELTA_RHO + ").",
    commandLineParamName = "rho",
    commandLineParamSynopsis = "-rho <double>",
    displayOrder = 0
  )
  public double getRho() {
    return backend.getRho();
  }

  public void setRho(double rho) {
    backend.setRho(rho);
  }

  @OptionMetadata(
    displayName = "epsilon",
    description = "The epsilon parameter (default = " + org.nd4j.linalg.learning.config.AdaDelta.DEFAULT_ADADELTA_EPSILON + ").",
    commandLineParamName = "epsilon",
    commandLineParamSynopsis = "-epsilon <double>",
    displayOrder = 1
  )
  public double getEpsilon() {
    return backend.getEpsilon();
  }

  public void setEpsilon(double epsilon) {
    backend.setEpsilon(epsilon);
  }

  /**
   * Set the learning rate
   *
   * @param learningRate Learning rate
   */
  @ProgrammaticProperty
  @Override
  public void setLearningRate(double learningRate) {
    log.warn("The method setLearningRate(double) has no effect on the AdaDelta updater since it "
        + "does not employ any learning rate.");
  }

  @ProgrammaticProperty
  @Override
  public double getLearningRate(){
    log.warn("The method getLearningRate() has no effect on the AdaDelta updater since it "
        + "does not employ any learning rate.");
    return 0.0;
  }

  @Override
  public void initializeBackend() {
    backend = new org.nd4j.linalg.learning.config.AdaDelta();
  }

  @Override
  public boolean hasLearningRate() {
    return false;
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  @Override
  public Enumeration<Option> listOptions() {
    return Option.listOptionsForClassHierarchy(this.getClass(),super.getClass()).elements();
  }

  /**
   * Gets the current settings of the Classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  @Override
  public String[] getOptions() {
    return Option.getOptionsForHierarchy(this, super.getClass());
  }

  /**
   * Parses a given list of options.
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    Option.setOptionsForHierarchy(options, this, super.getClass());
  }
}
