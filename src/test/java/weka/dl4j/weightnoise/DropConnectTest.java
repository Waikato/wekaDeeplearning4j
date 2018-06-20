package weka.dl4j.weightnoise;

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

public class DropConnectTest extends ApiWrapperTest<DropConnect> {

  @Test
  public void setWeightRetainProbability() {
    double value = 123.456;
    wrapper.setWeightRetainProbability(value);

    assertEquals(value, wrapper.getWeightRetainProbability(), PRECISION);
  }

  @Test
  public void setWeightRetainProbabilitySchedule() {
    for (Schedule sched : new Schedule[]{
        new ConstantSchedule(),
        new ExponentialSchedule(),
        new InverseSchedule(),
        new MapSchedule(),
        new PolySchedule(),
        new SigmoidSchedule(),
        new StepSchedule()
    }) {
      wrapper.setWeightRetainProbabilitySchedule(sched);

      assertEquals(sched, wrapper.getWeightRetainProbabilitySchedule());
    }
  }

  @Override
  public DropConnect getApiWrapper() {
    return new DropConnect();
  }
}