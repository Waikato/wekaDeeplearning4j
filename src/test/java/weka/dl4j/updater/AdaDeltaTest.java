package weka.dl4j.updater;

import static org.junit.Assert.*;

import org.junit.Test;

public class AdaDeltaTest extends AbstractUpdaterTest<AdaDelta> {

  @Test
  public void setRho() {
    double value = 123.456;
    wrapper.setRho(value);

    assertEquals(value, wrapper.getRho(), PRECISION);
  }

  @Test
  public void setEpsilon() {
    double value = 123.456;
    wrapper.setEpsilon(value);

    assertEquals(value, wrapper.getEpsilon(), PRECISION);
  }

  @Override
  public AdaDelta getApiWrapper() {
    return new AdaDelta();
  }
}