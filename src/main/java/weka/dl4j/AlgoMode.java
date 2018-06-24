package weka.dl4j;

import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;

/**
 * Proxy Enum for {@link org.deeplearning4j.nn.conf.ConvolutionMode}. This is necessary as
 * Weka's run script cannot find the enum classes during the option parsing as they reside in the
 * Dl4j backend and are at that time not visible to the class loader.
 *
 * @author Steven Lang
 */
public enum AlgoMode implements ApiWrapper<ConvolutionLayer.AlgoMode> {
  NO_WORKSPACE, PREFER_FASTEST, USER_SPECIFIED;

  @Override
  public ConvolutionLayer.AlgoMode getBackend() {
    return ConvolutionLayer.AlgoMode.valueOf(this.name());
  }

  @Override
  public void setBackend(ConvolutionLayer.AlgoMode newBackend) {
    // Do nothing as this enum does not have a state

  }

  @Override
  public void initializeBackend() {
    // Do nothing as this enum does not have a state
  }

  /**
   * Parse backend algo mode and return weka enum implementation.
   *
   * @param algoMode Convolution mode
   * @return Weka convolution mode enum implementation
   */
  public static AlgoMode fromBackend(
      ConvolutionLayer.AlgoMode algoMode) {
    return valueOf(algoMode.name());
  }
}
