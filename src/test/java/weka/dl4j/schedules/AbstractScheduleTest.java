package weka.dl4j.schedules;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import weka.dl4j.ApiWrapperTest;

public abstract class AbstractScheduleTest<T extends Schedule> extends ApiWrapperTest<T> {

  @Test
  public void setScheduleType() {
    for (ScheduleType type : ScheduleType.values()) {
      wrapper.setScheduleType(type);

      assertEquals(type, wrapper.getScheduleType());
    }
  }

  @Test
  public void setInitialValue() {
    double value = 123.456;
    wrapper.setInitialValue(value);

    assertEquals(value, wrapper.getInitialValue(), PRECISION);
  }

}
