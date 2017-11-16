package weka.dl4j.listener;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import weka.core.Option;
import weka.core.OptionHandler;

import java.util.Enumeration;

/**
 * Iteration listener that can be attached to a Dl4j model.
 *
 * @author Steven Lang
 */
public abstract class IterationListener
    implements org.deeplearning4j.optimize.api.IterationListener, OptionHandler {
  /** SerialVersionUID */
  private static final long serialVersionUID = 8106114790187499011L;

  /** Flag if already invoked */
  protected boolean invoked;

  /** Number of samples */
  protected int numSamples;
  /** Number of epochs */
  protected int numClasses;
  /** Number of classes */
  protected int numEpochs;
  /** Training dataset iterator */
  protected transient DataSetIterator validationIterator;
  /** Validation dataset iterator */
  protected transient DataSetIterator trainIterator;

  /**
   * Initialize the iterator with its necessary member variables
   * @param numClasses Number of classes
   * @param numEpochs Number of epochs
   * @param numSamples Number of Samples
   * @param trainIterator Training iterator
   * @param validationIterator Validation iterator
   */
  public void init(
      int numClasses,
      int numEpochs,
      int numSamples,
      DataSetIterator trainIterator,
      DataSetIterator validationIterator) {
    this.numClasses = numClasses;
    this.numEpochs = numEpochs;
    this.numSamples = numSamples;
    this.trainIterator = trainIterator;
    this.validationIterator = validationIterator;
  }

  /**
   * Log a message
   * @param msg Message
   */
  public abstract void log(String msg);

  @Override
  public boolean invoked() {
    return invoked;
  }

  @Override
  public void invoke() {
    this.invoked = true;
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  @Override
  public Enumeration<Option> listOptions() {

    return Option.listOptionsForClass(this.getClass()).elements();
  }

  /**
   * Gets the current settings of the Classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  @Override
  public String[] getOptions() {

    return Option.getOptions(this, this.getClass());
  }

  /**
   * Parses a given list of options.
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {

    Option.setOptions(options, this, this.getClass());
  }
}
