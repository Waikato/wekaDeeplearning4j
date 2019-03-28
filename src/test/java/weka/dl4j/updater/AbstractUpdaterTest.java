
package weka.dl4j.updater;

import static org.junit.Assert.assertEquals;

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

public abstract class AbstractUpdaterTest<T extends Updater> extends ApiWrapperTest<T> {

  @Test
  public void setLearningRate() {
    if (wrapper.hasLearningRate()) {
      double value = 123.456;
      wrapper.setLearningRate(value);

      assertEquals(value, wrapper.getLearningRate(), PRECISION);
    }
  }

  @Test
  public void setLearningRateSchedule() {
    for (Schedule sched : new Schedule[]{
        new ConstantSchedule(),
        new ExponentialSchedule(),
        new InverseSchedule(),
        new MapSchedule(),
        new PolySchedule(),
        new SigmoidSchedule(),
        new StepSchedule()
    }) {
      wrapper.setLearningRateSchedule(sched);

      assertEquals(sched, wrapper.getLearningRateSchedule());
    }
  }
}
