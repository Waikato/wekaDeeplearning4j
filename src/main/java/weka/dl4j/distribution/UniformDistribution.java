
package weka.dl4j.distribution;

import java.util.Enumeration;
import org.nd4j.shade.jackson.annotation.JsonTypeName;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;

/**
 * A version of DeepLearning4j's UniformDistribution that implements WEKA option handling.
 *
 * @author Eibe Frank
 * @author Steven Lang
 */
@JsonTypeName("uniform")
public class UniformDistribution extends
    Distribution<org.deeplearning4j.nn.conf.distribution.UniformDistribution> implements
    OptionHandler {


  private static final long serialVersionUID = -822639410912329551L;

  @OptionMetadata(
      displayName = "lower limit",
      description = "The lower limit (default = -1.0).",
      commandLineParamName = "lower",
      commandLineParamSynopsis = "-lower <double>",
      displayOrder = 1
  )
  public double getLower() {
    return backend.getLower();
  }

  public void setLower(double mean) {
    backend.setLower(mean);
  }

  @OptionMetadata(
      displayName = "upper limit",
      description = "The upper limit (default = 1.0).",
      commandLineParamName = "upper",
      commandLineParamSynopsis = "-upper <double>",
      displayOrder = 2
  )
  public double getUpper() {
    return backend.getUpper();
  }

  public void setUpper(double std) {
    backend.setUpper(std);
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
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {

    Option.setOptions(options, this, this.getClass());
  }

  @Override
  public void initializeBackend() {
    // Constructions normal distribution with lower limit -1 and upper limit 1
    backend = new org.deeplearning4j.nn.conf.distribution.UniformDistribution(-1.0, 1.0);
  }
}
