
package weka.dl4j.dropout;

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

public class AlphaDropoutTest extends ApiWrapperTest<AlphaDropout> {

  @Test
  public void setpSchedule() {
    for (Schedule sched : new Schedule[]{
        new ConstantSchedule(),
        new ExponentialSchedule(),
        new InverseSchedule(),
        new MapSchedule(),
        new PolySchedule(),
        new SigmoidSchedule(),
        new StepSchedule()
    }) {
      wrapper.setpSchedule(sched);

      assertEquals(sched, wrapper.getpSchedule());
    }
  }

  @Test
  public void setP() {
    double value = 123.456;
    wrapper.setP(value);

    assertEquals(value, wrapper.getP(), PRECISION);
  }

  @Test
  public void setAlpha() {
    double value = 123.456;
    wrapper.setAlpha(value);

    assertEquals(value, wrapper.getAlpha(), PRECISION);
  }

  @Test
  public void setLambda() {
    double value = 123.456;
    wrapper.setLambda(value);

    assertEquals(value, wrapper.getLambda(), PRECISION);
  }

  @Override
  public AlphaDropout getApiWrapper() {
    return new AlphaDropout();
  }
}
