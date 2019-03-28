
package weka.dl4j.text.tokenization.preprocessor;

import java.io.Serializable;
import java.util.Enumeration;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.core.stemmers.NullStemmer;
import weka.core.stemmers.Stemmer;
import weka.dl4j.text.tokenization.preprocessor.impl.StemmingPreProcessorImpl;

/**
 * A wrapper that extends the PreProcessor API for {@link StemmingPreProcessorImpl}.
 *
 * @author Steven Lang
 */
public class StemmingPreProcessor
    extends TokenPreProcess<StemmingPreProcessorImpl>
    implements Serializable, OptionHandler {

  /**
   * For serialization
   */
  private static final long serialVersionUID = 436336311776463684L;

  /**
   * A Weka stemmer objet
   */
  private Stemmer stemmer = new NullStemmer();

  /**
   * Returns a string describing this object.
   *
   * @return a description of the object suitable for displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return "This tokenizer preprocessor implements basic cleaning inherited from CommonPreProcessor + does stemming using a Weka Stemmer.\n";
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

  @OptionMetadata(
      displayName = "stemmer",
      description = "The Weka stemmer to use.",
      commandLineParamName = "stemmer",
      commandLineParamSynopsis = "-stemmer <String>",
      displayOrder = 0
  )
  public Stemmer getStemmer() {
    return stemmer;
  }

  public void setStemmer(Stemmer stemmer) {
    this.stemmer = stemmer;
  }

  @Override
  public void initializeBackend() {
    backend = new StemmingPreProcessorImpl();
  }
}
