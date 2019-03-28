
package weka.dl4j.activations;

import org.nd4j.shade.jackson.annotation.JsonTypeName;
import weka.core.Option;
import weka.core.OptionHandler;

import java.util.Enumeration;

/**
 * A version of DeepLearning4j's ActivationHardSigmoid that implements WEKA option handling.
 *
 * @author Eibe Frank
 */
@JsonTypeName("HardSigmoid")
public class ActivationHardSigmoid extends Activation<org.nd4j.linalg.activations.impl.ActivationHardSigmoid>
    implements OptionHandler {

  private static final long serialVersionUID = -1571785817480969424L;
  @Override
  public void initializeBackend() {
    backend = new org.nd4j.linalg.activations.impl.ActivationHardSigmoid();
  }
  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  @Override
  public Enumeration<Option> listOptions() {

    return Option.listOptionsForClass(this.getClass()).elements();
  }

  /**
   * Gets the current settings of the Classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  @Override
  public String[] getOptions() {

    return Option.getOptions(this, this.getClass());
  }

  /**
   * Parses a given list of options.
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {

    Option.setOptions(options, this, this.getClass());
  }
}
