
package weka.dl4j.text.tokenization.preprocessor;

import java.io.Serializable;
import java.util.Enumeration;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.dl4j.text.tokenization.preprocessor.impl.EndingPreProcessorImpl;

/**
 * A wrapper that extends the PreProcessor API for {@link EndingPreProcessorImpl}.
 *
 * @author Steven Lang
 */
public class EndingPreProcessor
    extends TokenPreProcess<EndingPreProcessorImpl>
    implements Serializable, OptionHandler {

  /**
   * For Serialization
   */
  private static final long serialVersionUID = -7863874149371478868L;

  /**
   * Returns a string describing this object.
   *
   * @return a description of the object suitable for displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return "Gets rid of endings: ed,ing, ly, s, ..\n";
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
    backend = new EndingPreProcessorImpl();
  }
}
