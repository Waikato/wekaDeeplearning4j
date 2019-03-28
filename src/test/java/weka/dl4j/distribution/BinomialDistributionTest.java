
package weka.dl4j.distribution;

import static org.junit.Assert.*;

import org.junit.Test;
import weka.dl4j.ApiWrapperTest;

public class BinomialDistributionTest extends ApiWrapperTest<BinomialDistribution> {

  @Test
  public void setProbabilityOfSuccess() {
    double prob = 123.456;
    wrapper.setProbabilityOfSuccess(prob);

    assertEquals(prob, wrapper.getProbabilityOfSuccess(), PRECISION);
  }

  @Test
  public void setNumberOfTrials() {
    int num = 123;
    wrapper.setNumberOfTrials(num);

    assertEquals(num, wrapper.getNumberOfTrials(), PRECISION);

  }

  @Override
  public BinomialDistribution getApiWrapper() {
    return new  BinomialDistribution();
  }
}
