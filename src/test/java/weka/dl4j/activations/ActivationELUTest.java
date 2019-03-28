
package weka.dl4j.activations;

import static org.junit.Assert.*;

import org.junit.Test;
import weka.dl4j.ApiWrapperTest;

public class ActivationELUTest extends ApiWrapperTest<ActivationELU> {

  @Test
  public void setAlpha() {
    double alpha = 123.456;
    wrapper.setAlpha(alpha);

    assertEquals(alpha, wrapper.getAlpha(), PRECISION);
  }

  @Override
  public ActivationELU getApiWrapper() {
    return new  ActivationELU();
  }
}
