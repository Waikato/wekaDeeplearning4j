
package weka.dl4j.updater;

import java.util.Enumeration;
import weka.core.Option;
import weka.core.OptionMetadata;

/**
 * A WEKA version of DeepLearning4j's AdaGrad.
 *
 * @author Steven Lang
 */
public class AdaGrad extends Updater<org.nd4j.linalg.learning.config.AdaGrad> {
  private static final long serialVersionUID = 3881105990718165790L;

  @OptionMetadata(
    displayName = "epsilon",
    description =
        "The epsilon parameter (default = "
            + org.nd4j.linalg.learning.config.AdaGrad.DEFAULT_ADAGRAD_EPSILON
            + ").",
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

  @Override
  public void initializeBackend() {
    backend = new org.nd4j.linalg.learning.config.AdaGrad();
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
