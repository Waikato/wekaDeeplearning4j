package weka.dl4j.updater;

import static org.nd4j.linalg.learning.config.Nesterovs.DEFAULT_NESTEROV_MOMENTUM;

import java.util.Enumeration;
import weka.core.Option;
import weka.core.OptionMetadata;
import weka.gui.ProgrammaticProperty;

/**
 * A WEKA version of DeepLearning4j's Nesterovs.
 *
 * @author Steven Lang
 */
public class Nesterovs extends  Updater<org.nd4j.linalg.learning.config.Nesterovs> {
  private static final long serialVersionUID = 927121528229628203L;

  @OptionMetadata(
    displayName = "momentum",
    description = "The momentum (default = " + DEFAULT_NESTEROV_MOMENTUM + ").",
    commandLineParamName = "momentum",
    commandLineParamSynopsis = "-momentum <double>",
    displayOrder = 1
  )
  public double getMomentum() {
    return backend.getMomentum();
  }

  public void setMomentum(double momentum) {
    backend.setMomentum(momentum);
  }

  @Override
  public void initializeBackend() {
    backend = new org.nd4j.linalg.learning.config.Nesterovs();
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
