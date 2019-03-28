
package weka.dl4j.distribution;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import weka.dl4j.ApiWrapperTest;

public class TruncatedNormalDistributionTest extends ApiWrapperTest<TruncatedNormalDistribution> {

  @Test
  public void setMean() {
    double value = 123.456;
    wrapper.setMean(value);

    assertEquals(value, wrapper.getMean(), PRECISION);
  }

  @Test
  public void setStd() {
    double value = 123.456;
    wrapper.setStd(value);

    assertEquals(value, wrapper.getStd(), PRECISION);
  }

  @Override
  public TruncatedNormalDistribution getApiWrapper() {
    return new TruncatedNormalDistribution();
  }
}
