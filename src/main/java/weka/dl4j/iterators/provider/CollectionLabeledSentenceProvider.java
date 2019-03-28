
package weka.dl4j.iterators.provider;

import java.util.List;

/**
 * Extend CollectionLabeledSentenceProvider from DL4J to support setting numClasses dynamically for
 * regression tasks.
 *
 * @author Steven Lang
 */
public class CollectionLabeledSentenceProvider
    extends org.deeplearning4j.iterator.provider.CollectionLabeledSentenceProvider {

  /**
   * Number of classes
   */
  private final int numClasses;

  public CollectionLabeledSentenceProvider(
      List<String> sentences, List<String> labelsForSentences, int numClasses) {
    super(sentences, labelsForSentences, null);
    this.numClasses = numClasses;
  }

  @Override
  public int numLabelClasses() {
    return numClasses;
  }
}
