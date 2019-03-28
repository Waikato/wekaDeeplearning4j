
package weka.dl4j;

/**
 * Proxy Enum for {@link org.deeplearning4j.nn.conf.layers.PoolingType}. This is necessary as Weka's
 * run script cannot find the enum classes during the option parsing as they reside in the Dl4j
 * backend and are at that time not visible to the class loader.
 *
 * @author Steven Lang
 */
public enum PoolingType implements ApiWrapper<org.deeplearning4j.nn.conf.layers.PoolingType> {
  MAX, AVG, SUM, PNORM;

  /**
   * Parse backend pooling type and return weka enum implementation.
   *
   * @param poolingType Pooling type
   * @return Weka pooling type enum implementation
   */
  public static PoolingType fromBackend(
      org.deeplearning4j.nn.conf.layers.PoolingType poolingType) {
    return valueOf(poolingType.name());
  }

  @Override
  public org.deeplearning4j.nn.conf.layers.PoolingType getBackend() {
    return org.deeplearning4j.nn.conf.layers.PoolingType.valueOf(this.name());
  }

  @Override
  public void setBackend(org.deeplearning4j.nn.conf.layers.PoolingType newBackend) {
    // Do nothing as this enum does not have a state

  }

  @Override
  public void initializeBackend() {
    // Do nothing as this enum does not have a state
  }
}
