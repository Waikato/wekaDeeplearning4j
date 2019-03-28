
package weka.dl4j.layers;

import java.io.Serializable;
import java.util.Enumeration;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.dl4j.activations.ActivationSoftmax;
import weka.dl4j.lossfunctions.LossFunction;
import weka.dl4j.lossfunctions.LossMCXENT;

/**
 * A version of DeepLearning4j's RnnOutputLayer that implements WEKA option handling.
 *
 * @author Steven Lang
 */
public class RnnOutputLayer extends
    FeedForwardLayer<org.deeplearning4j.nn.conf.layers.RnnOutputLayer>
    implements OptionHandler, Serializable {

  /**
   * SerialVersionUID
   */
  private static final long serialVersionUID = 6333123285853131216L;

  /**
   * Constructor for setting some defaults.
   */
  public RnnOutputLayer() {
    super();
    setLayerName("RnnOutput layer");
    setActivationFunction(new ActivationSoftmax());
    setLossFn(new LossMCXENT());
  }

  @Override
  public void initializeBackend() {
    backend = new org.deeplearning4j.nn.conf.layers.RnnOutputLayer();
  }

  /**
   * Global info.
   *
   * @return string describing this class.
   */
  public String globalInfo() {
    return "A RnnOutput layer from DeepLearning4J.";
  }


  @OptionMetadata(
      displayName = "loss function",
      description = "The loss function to use (default = LossMCXENT).",
      commandLineParamName = "lossFn",
      commandLineParamSynopsis = "-lossFn <specification>",
      displayOrder = 1
  )
  public LossFunction<? extends ILossFunction> getLossFn() {
    return LossFunction.create(backend.getLossFn());
  }

  public void setLossFn(LossFunction<? extends ILossFunction> lossFn) {
    backend.setLossFn(lossFn.getBackend());
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
