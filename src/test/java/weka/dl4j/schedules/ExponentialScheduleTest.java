
package weka.dl4j.schedules;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class ExponentialScheduleTest extends AbstractScheduleTest<ExponentialSchedule> {

  @Test
  public void setGamma() {
    double value = 123.456;
    wrapper.setGamma(value);
    assertEquals(value, wrapper.getGamma(), PRECISION);
  }

  @Override
  public ExponentialSchedule getApiWrapper() {
    return new ExponentialSchedule();
  }
}
