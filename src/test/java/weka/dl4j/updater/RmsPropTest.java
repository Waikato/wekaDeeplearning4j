package weka.dl4j.updater;

import static org.junit.Assert.*;

import org.junit.Test;

public class RmsPropTest extends AbstractUpdaterTest<RmsProp> {

  @Test
  public void setRmsDecay() {
    double value = 123.456;
    wrapper.setRmsDecay(value);

    assertEquals(value, wrapper.getRmsDecay(), PRECISION);
  }

  @Test
  public void setEpsilon() {
    double value = 123.456;
    wrapper.setEpsilon(value);

    assertEquals(value, wrapper.getEpsilon(), PRECISION);
  }

  @Override
  public RmsProp getApiWrapper() {
    return new RmsProp();
  }
}