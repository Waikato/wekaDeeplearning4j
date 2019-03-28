
package weka.dl4j.text.stopwords;

import java.util.ArrayList;
import java.util.List;

/**
 * <!-- globalinfo-start --> Dummy stopwords scheme, returns an empty list of stopWords.. <!--
 * globalinfo-end -->
 *
 * @author fracpete, Felipe Bravo-Marquez
 */
public class Dl4jNull extends Dl4jAbstractStopwords {

  /**
   * For serialization.
   */
  private static final long serialVersionUID = -9129283649432847013L;

  /**
   * The list of stopwords.
   */
  protected List<String> stopWords;

  /* (non-Javadoc)
   * @see weka.dl4j.text.stopwords.Dl4jAbstractStopwords#getStopList()
   */
  @Override
  public List<String> getStopList() {

    return stopWords;
  }

  /* (non-Javadoc)
   * @see weka.dl4j.text.stopwords.Dl4jAbstractStopwords#initialize()
   */
  @Override
  public void initialize() {
    stopWords = new ArrayList<String>();
  }

  /**
   * Returns a string describing the stopwords scheme.
   *
   * @return a description suitable for displaying in the gui
   */
  @Override
  public String globalInfo() {
    return "Dummy stopwords scheme, returns an empty list of stopWords.";
  }

  /**
   * Returns true if the given string is a stop word.
   *
   * @param word the word to test
   * @return always false
   */
  @Override
  protected boolean is(String word) {
    return false;
  }
}
