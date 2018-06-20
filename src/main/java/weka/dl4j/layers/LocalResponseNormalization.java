package weka.dl4j.layers;

import java.util.Enumeration;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import weka.core.Option;
import weka.core.OptionMetadata;

/**
 * A version of DeepLearning4j's LocalResponseNormalization layer that implements WEKA option handling.
 *
 * @author Steven Lang
 */
public class LocalResponseNormalization extends
    Layer<org.deeplearning4j.nn.conf.layers.LocalResponseNormalization> {

  private static final long serialVersionUID = 1846770958541767815L;

  /** Constructor for setting some defaults. */
  public LocalResponseNormalization() {
    super();
    setLayerName("LocalResponseNormalization layer");
  }

  @Override
  public void initializeBackend() {
    backend = new org.deeplearning4j.nn.conf.layers.LocalResponseNormalization();
  }


  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  @Override
  public Enumeration<Option> listOptions() {
    return Option.listOptionsForClassHierarchy(this.getClass(), super.getClass()).elements();
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
