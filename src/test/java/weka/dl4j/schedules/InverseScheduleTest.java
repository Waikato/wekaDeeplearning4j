
package weka.dl4j.schedules;

import static org.junit.Assert.*;

import org.junit.Test;
import weka.dl4j.ApiWrapperTest;

public class InverseScheduleTest extends AbstractScheduleTest<InverseSchedule> {

  @Test
  public void setGamma() {
    double value = 123.456;
    wrapper.setGamma(value);

    assertEquals(value, wrapper.getGamma(), PRECISION);

  }

  @Test
  public void setPower() {
    double value = 123.456;
    wrapper.setPower(value);

    assertEquals(value, wrapper.getPower(), PRECISION);
  }

  @Override
  public InverseSchedule getApiWrapper() {
    return new InverseSchedule();
  }
}
