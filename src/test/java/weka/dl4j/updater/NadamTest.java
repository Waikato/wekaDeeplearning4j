
package weka.dl4j.updater;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class NadamTest extends AbstractUpdaterTest<Nadam> {

  @Test
  public void setBeta1() {
    double value = 123.456;
    wrapper.setBeta1(value);

    assertEquals(value, wrapper.getBeta1(), PRECISION);
  }

  @Test
  public void setBeta2() {
    double value = 123.456;
    wrapper.setBeta2(value);

    assertEquals(value, wrapper.getBeta2(), PRECISION);
  }

  @Test
  public void setEpsilon() {
    double value = 123.456;
    wrapper.setEpsilon(value);

    assertEquals(value, wrapper.getEpsilon(), PRECISION);
  }

  @Override
  public Nadam getApiWrapper() {
    return new Nadam();
  }
}
