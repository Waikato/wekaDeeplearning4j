
package weka.dl4j.text.tokenization.tokenizer.factory;

import java.io.Serializable;
import java.util.Enumeration;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.dl4j.text.tokenization.tokenizer.factory.impl.DefaultTokenizerFactoryImpl;

/**
 * A wrapper that extends the TokenizerFactory API for {@link DefaultTokenizerFactoryImpl}.
 *
 * @author Steven Lang
 */
public class DefaultTokenizerFactory
    extends
    TokenizerFactory<DefaultTokenizerFactoryImpl>
    implements Serializable, OptionHandler {

  /**
   * For Serialization
   */
  private static final long serialVersionUID = 762619182220458182L;

  /**
   * Returns a string describing this object.
   *
   * @return a description of the object suitable for displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return "Default tokenizer based on string tokenizer.";
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
    backend = new DefaultTokenizerFactoryImpl();
  }
}
