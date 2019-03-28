
package weka.dl4j.distribution;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import weka.dl4j.ApiWrapperTest;

public class UniformDistributionTest extends ApiWrapperTest<UniformDistribution> {

  @Test
  public void setLower() {
    double value = 123.456;
    wrapper.setLower(value);

    assertEquals(value, wrapper.getLower(), PRECISION);
  }

  @Test
  public void setUpper() {
    double value = 123.456;
    wrapper.setUpper(value);

    assertEquals(value, wrapper.getUpper(), PRECISION);
  }

  @Override
  public UniformDistribution getApiWrapper() {
    return new UniformDistribution();
  }
}
