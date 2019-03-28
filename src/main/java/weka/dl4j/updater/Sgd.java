
package weka.dl4j.updater;

import java.util.Enumeration;
import weka.core.Option;

/**
 * A WEKA version of DeepLearning4j's Sgd.
 *
 * @author Steven Lang
 */
public class Sgd extends Updater<org.nd4j.linalg.learning.config.Sgd> {

  private static final long serialVersionUID = 1852959048173443658L;

  @Override
  public void initializeBackend() {
    backend = new org.nd4j.linalg.learning.config.Sgd();
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
