package weka.dl4j.schedules;

import static org.junit.Assert.*;

import org.junit.Test;
import weka.dl4j.ApiWrapperTest;

public class StepScheduleTest extends AbstractScheduleTest<StepSchedule> {

  @Test
  public void setDecayRate() {
    double value = 123.456;
    wrapper.setDecayRate(value);

    assertEquals(value, wrapper.getDecayRate(), PRECISION);
  }

  @Test
  public void setStep() {
    double value = 123.456;
    wrapper.setStep(value);

    assertEquals(value, wrapper.getStep(), PRECISION);

  }

  @Override
  public StepSchedule getApiWrapper() {
    return new StepSchedule();
  }
}