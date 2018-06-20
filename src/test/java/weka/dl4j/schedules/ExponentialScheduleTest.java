package weka.dl4j.schedules;

import static org.junit.Assert.*;

import org.junit.Test;
import weka.dl4j.ApiWrapperTest;

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