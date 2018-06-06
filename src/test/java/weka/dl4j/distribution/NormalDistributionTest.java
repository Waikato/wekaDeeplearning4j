package weka.dl4j.distribution;

import static org.junit.Assert.*;

import org.junit.Test;
import weka.dl4j.ApiWrapperTest;

public class NormalDistributionTest extends ApiWrapperTest<NormalDistribution> {

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
  public NormalDistribution getApiWrapper() {
    return new  NormalDistribution();
  }
}