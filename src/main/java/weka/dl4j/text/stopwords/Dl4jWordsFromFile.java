
package weka.dl4j.text.stopwords;

import java.util.ArrayList;
import java.util.List;

/**
 * @author fracpete, Felipe Bravo-Marquez
 */
public class Dl4jWordsFromFile extends Dl4jAbstractFileBasedStopwords {

  /**
   * for serialization.
   */
  private static final long serialVersionUID = -722795295494945193L;

  /**
   * The list of stopwords.
   */
  protected List<String> stopWords;

  /**
   * Returns a string describing the stopwords scheme.
   *
   * @return a description suitable for displaying in the gui
   */
  @Override
  public String globalInfo() {
    return "Uses the stopwords located in the specified file (ignored _if "
        + "pointing to a directory). One stopword per line. Lines "
        + "starting with '#' are considered comments and ignored.";
  }

  /**
   * Returns the tip text for this property.
   *
   * @return tip text for this property suitable for displaying in the explorer/experimenter gui
   */
  @Override
  public String stopwordsTipText() {
    return "The file containing the stopwords.";
  }

  /**
   * Performs intialization of the scheme.
   */
  @Override
  public void initialize() {
    List<String> words;

    stopWords = new ArrayList<String>();
    words = read();
    for (String word : words) {
      // comment?
      if (!word.startsWith("#")) {
        stopWords.add(word);
      }
    }
  }

  /**
   * Returns true if the given string is a stop word.
   *
   * @param word the word to test
   * @return true if the word is a stopword
   */
  @Override
  protected synchronized boolean is(String word) {
    return stopWords.contains(word.trim().toLowerCase());
  }

  /* (non-Javadoc)
   * @see weka.dl4j.text.stopwords.Dl4jAbstractStopwords#getStopList()
   */
  @Override
  public List<String> getStopList() {
    return stopWords;
  }
}
