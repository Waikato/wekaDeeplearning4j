
package weka.dl4j.activations;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import weka.dl4j.ApiWrapperTest;

public class ActivationLReLUTest extends ApiWrapperTest<ActivationLReLU> {

  @Test
  public void setAlpha() {
    double alpha = 123.456;
    wrapper.setAlpha(alpha);

    assertEquals(alpha, wrapper.getAlpha(), PRECISION);
  }

  @Override
  public ActivationLReLU getApiWrapper() {
    return new ActivationLReLU();
  }
}
