
package weka.dl4j.layers;

import weka.core.Option;
import weka.core.OptionHandler;
import weka.dl4j.activations.ActivationReLU;

import java.io.Serializable;
import java.util.Enumeration;

/**
 * A version of DeepLearning4j's DenseLayer that implements WEKA option handling.
 *
 * @author Christopher Beckham
 * @author Eibe Frank
 * @author Steven Lang
 */
public class DenseLayer extends FeedForwardLayer<org.deeplearning4j.nn.conf.layers.DenseLayer>
    implements OptionHandler, Serializable {

  // The serial version ID used when serializing this class
  protected static final long serialVersionUID = -6905917800811990400L;

  /** Constructor for setting some defaults. */
  public DenseLayer() {
    super();
    setLayerName("Dense layer");
    setActivationFunction(new ActivationReLU());
  }

  @Override
  public void initializeBackend() {
    backend= new org.deeplearning4j.nn.conf.layers.DenseLayer();
  }

  /**
   * Global info.
   *
   * @return string describing this class.
   */
  public String globalInfo() {
    return "A densely connected layer from DeepLearning4J.";
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
}
