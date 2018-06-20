package weka.dl4j.activations;

import static org.junit.Assert.*;

import org.junit.Test;
import weka.dl4j.ApiWrapperTest;

public class ActivationRReLUTest extends ApiWrapperTest<ActivationRReLU> {

  @Test
  public void setLowerBound() {
    double bound = 123.456;
    wrapper.setLowerBound(bound);

    assertEquals(bound, wrapper.getLowerBound(), PRECISION);
  }

  @Test
  public void setUpperBound() {
    double bound = 123.456;
    wrapper.setUpperBound(bound);

    assertEquals(bound, wrapper.getUpperBound(), PRECISION);
  }

  @Override
  public ActivationRReLU getApiWrapper() {
    return new  ActivationRReLU();
  }
}