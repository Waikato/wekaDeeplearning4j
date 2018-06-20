package weka.dl4j.distribution;

import static org.junit.Assert.*;

import org.junit.Test;
import weka.dl4j.ApiWrapperTest;

public class OrthogonalDistributionTest extends ApiWrapperTest<OrthogonalDistribution> {


  @Test
  public void setGain() {
    double value = 123.456;
    wrapper.setGain(value);

    assertEquals(value, wrapper.getGain(), PRECISION);
  }

  @Override
  public OrthogonalDistribution getApiWrapper() {
    return new  OrthogonalDistribution();
  }
}