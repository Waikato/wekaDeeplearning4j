
package weka.dl4j.iterators.instance.sequence;

import weka.dl4j.iterators.instance.AbstractInstanceIterator;

/**
 * Marker class to differentiate between iterators for
 * {@link weka.classifiers.functions.Dl4jMlpClassifier} and
 * {@link weka.classifiers.functions.RnnSequenceClassifier}.
 *
 * @author Steven Lang
 */
public abstract class AbstractSequenceInstanceIterator extends AbstractInstanceIterator{
  private static final long serialVersionUID = 6449591540279265588L;
}
