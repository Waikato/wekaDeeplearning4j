package weka.dl4j.updater;

import static org.junit.Assert.*;

import org.junit.Test;

public class AdaGradTest extends AbstractUpdaterTest<AdaGrad> {

  @Test
  public void setEpsilon() {
    double value = 123.456;
    wrapper.setEpsilon(value);

    assertEquals(value, wrapper.getEpsilon(), PRECISION);
  }

  @Override
  public AdaGrad getApiWrapper() {
    return new AdaGrad();
  }
}