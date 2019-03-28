
package weka.dl4j.text.stopwords;

import java.util.List;
import weka.core.stopwords.AbstractStopwords;

/**
 * Abstract stopwords handler for DL4j.
 *
 * @author Felipe Bravo-Marquez
 */
public abstract class Dl4jAbstractStopwords extends AbstractStopwords {

  /**
   * for serialization
   */
  private static final long serialVersionUID = -2167994358835350653L;

  /**
   * Returns the list of stopwords.
   *
   * @return the list of stopwords
   */
  public abstract List<String> getStopList();

  /**
   * initializes the dictionary
   */
  public abstract void initialize();
}
