package weka.dl4j.schedules;

import static org.junit.Assert.*;

import org.junit.Test;
import weka.dl4j.ApiWrapperTest;

public class PolyScheduleTest extends AbstractScheduleTest<PolySchedule> {

  @Test
  public void setMaxIter() {
    int value = 123;
    wrapper.setMaxIter(value);

    assertEquals(value, wrapper.getMaxIter(), PRECISION);
  }

  @Test
  public void setPower() {
    double value = 123.456;
    wrapper.setPower(value);

    assertEquals(value, wrapper.getPower(), PRECISION);
  }

  @Override
  public PolySchedule getApiWrapper() {
    return new PolySchedule();
  }
}