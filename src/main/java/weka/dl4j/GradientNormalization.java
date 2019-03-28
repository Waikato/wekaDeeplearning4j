
package weka.dl4j;

/**
 * Proxy Enum for {@link org.deeplearning4j.nn.conf.GradientNormalization}. This is necessary as
 * Weka's run script cannot find the enum classes during the option parsing as they reside in the
 * Dl4j backend and are at that time not visible to the class loader.
 *
 * @author Steven Lang
 */
public enum GradientNormalization implements
    ApiWrapper<org.deeplearning4j.nn.conf.GradientNormalization> {
  None,
  RenormalizeL2PerLayer,
  RenormalizeL2PerParamType,
  ClipElementWiseAbsoluteValue,
  ClipL2PerLayer,
  ClipL2PerParamType;

  public org.deeplearning4j.nn.conf.GradientNormalization getBackend() {
    return org.deeplearning4j.nn.conf.GradientNormalization.valueOf(this.name());
  }

  @Override
  public void setBackend(org.deeplearning4j.nn.conf.GradientNormalization newBackend) {
    // Do nothing as this enum does not have a state
  }

  @Override
  public void initializeBackend() {
    // Do nothing as this enum does not have a state
  }

  /**
   * Parse backend gradient normalization and return weka enum implementation.
   *
   * @param gradientNormalization GradientNormalization
   * @return Weka gradient normalization enum implementation
   */
  public static GradientNormalization fromBackend(
      org.deeplearning4j.nn.conf.GradientNormalization gradientNormalization) {
    return valueOf(gradientNormalization.name());
  }
}
