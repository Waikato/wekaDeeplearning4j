
package weka.dl4j.text.tokenization.preprocessor;

import java.io.Serializable;
import java.util.Enumeration;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.dl4j.text.tokenization.preprocessor.impl.CommonPreProcessorImpl;

/**
 * A wrapper that extends the PreProcessor API for {@link CommonPreProcessorImpl}.
 *
 * @author Steven Lang
 */
public class CommonPreProcessor
    extends TokenPreProcess<CommonPreProcessorImpl>
    implements Serializable, OptionHandler {

  /**
   * For serialization
   */
  private static final long serialVersionUID = 7854676262098995012L;

  /**
   * Returns a string describing this object.
   *
   * @return a description of the object suitable for displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return "All numbers, punctuation symbols and some special symbols are stripped off. \n"
        + "Additionally it forces lower case for all tokens.\n";
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

  @Override
  public void initializeBackend() {
    backend = new CommonPreProcessorImpl();
  }
}
