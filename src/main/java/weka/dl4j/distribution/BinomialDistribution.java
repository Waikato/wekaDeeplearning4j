
package weka.dl4j.distribution;

import java.util.Enumeration;
import org.nd4j.shade.jackson.annotation.JsonTypeName;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;

/**
 * A version of DeepLearning4j's BinomialDistribution that implements WEKA option handling.
 * Currently only allows one trial.
 *
 * @author Eibe Frank
 * @author Steven Lang
 */
@JsonTypeName("binomial")
public class BinomialDistribution
    extends Distribution<org.deeplearning4j.nn.conf.distribution.BinomialDistribution>
    implements OptionHandler {

  private static final long serialVersionUID = 4200787281772886115L;

  protected int numberOfTrials = 1;

  @OptionMetadata(
    displayName = "probability of success",
    description = "The probability of success (default = 0.5).",
    commandLineParamName = "prob",
    commandLineParamSynopsis = "-prob <double>",
    displayOrder = 1
  )
  public double getProbabilityOfSuccess() {
    return backend.getProbabilityOfSuccess();
  }

  public void setProbabilityOfSuccess(double probabilityOfSuccess) {
    backend.setProbabilityOfSuccess(probabilityOfSuccess);
  }

  @OptionMetadata(
    displayName = "number of trials",
    description = "The number of trials (default = 1).",
    commandLineParamName = "n",
    commandLineParamSynopsis = "-n <int>",
    displayOrder = 1
  )
  public double getNumberOfTrials() {
    return numberOfTrials;
  }

  public void setNumberOfTrials(int numberOfTrials) {
    this.numberOfTrials = numberOfTrials;
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

  @Override
  public void initializeBackend() {
    // Constructs binomial distribution with 1 trial and success probability 0.5
    backend = new org.deeplearning4j.nn.conf.distribution.BinomialDistribution(numberOfTrials, 0.5);
  }

  @Override
  public org.deeplearning4j.nn.conf.distribution.BinomialDistribution getBackend() {
    return new org.deeplearning4j.nn.conf.distribution.BinomialDistribution(numberOfTrials, backend.getProbabilityOfSuccess());
  }
}
