
package weka.dl4j;

/**
 * Proxy Enum for {@link org.deeplearning4j.nn.conf.ConvolutionMode}. This is necessary as Weka's
 * run script cannot find the enum classes during the option parsing as they reside in the Dl4j
 * backend and are at that time not visible to the class loader.
 *
 * @author Steven Lang
 */
public enum ConvolutionMode implements ApiWrapper<org.deeplearning4j.nn.conf.ConvolutionMode> {
  Strict, Truncate, Same;

  /**
   * Parse backend convolution mode and return weka enum implementation.
   *
   * @param convolutionMode Convolution mode
   * @return Weka convolution mode enum implementation
   */
  public static ConvolutionMode fromBackend(
      org.deeplearning4j.nn.conf.ConvolutionMode convolutionMode) {
    return valueOf(convolutionMode.name());
  }

  @Override
  public org.deeplearning4j.nn.conf.ConvolutionMode getBackend() {
    return org.deeplearning4j.nn.conf.ConvolutionMode.valueOf(this.name());
  }

  @Override
  public void setBackend(org.deeplearning4j.nn.conf.ConvolutionMode newBackend) {
    // Do nothing as this enum does not have a state

  }

  @Override
  public void initializeBackend() {
    // Do nothing as this enum does not have a state
  }
}
