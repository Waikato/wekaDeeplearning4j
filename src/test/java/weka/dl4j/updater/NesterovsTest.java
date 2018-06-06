package weka.dl4j.updater;

import static org.junit.Assert.*;

import org.junit.Test;

public class NesterovsTest extends AbstractUpdaterTest<Nesterovs> {

  @Test
  public void setMomentum() {
    double value = 123.456;
    wrapper.setMomentum(value);

    assertEquals(value, wrapper.getMomentum(), PRECISION);
  }

  @Override
  public Nesterovs getApiWrapper() {
    return new Nesterovs();
  }
}