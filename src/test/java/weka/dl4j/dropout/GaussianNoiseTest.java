
package weka.dl4j.dropout;

import static org.junit.Assert.*;

import org.junit.Test;
import weka.dl4j.ApiWrapperTest;
import weka.dl4j.schedules.ConstantSchedule;
import weka.dl4j.schedules.ExponentialSchedule;
import weka.dl4j.schedules.InverseSchedule;
import weka.dl4j.schedules.MapSchedule;
import weka.dl4j.schedules.PolySchedule;
import weka.dl4j.schedules.Schedule;
import weka.dl4j.schedules.SigmoidSchedule;
import weka.dl4j.schedules.StepSchedule;

public class GaussianNoiseTest extends ApiWrapperTest<GaussianNoise> {

  @Test
  public void setStdDev() {
    double value = 123.456;
    wrapper.setStdDev(value);

    assertEquals(value, wrapper.getStdDev(), PRECISION);
  }

  @Test
  public void setRateSchedule() {
    for (Schedule sched : new Schedule[]{
        new ConstantSchedule(),
        new ExponentialSchedule(),
        new InverseSchedule(),
        new MapSchedule(),
        new PolySchedule(),
        new SigmoidSchedule(),
        new StepSchedule()
    }) {
      wrapper.setRateSchedule(sched);

      assertEquals(sched, wrapper.getRateSchedule());
    }
  }

  @Override
  public GaussianNoise getApiWrapper() {
    return new GaussianNoise();
  }
}
