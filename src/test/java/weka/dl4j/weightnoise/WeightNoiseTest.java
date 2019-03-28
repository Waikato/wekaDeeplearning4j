
package weka.dl4j.weightnoise;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;
import weka.dl4j.ApiWrapperTest;
import weka.dl4j.distribution.BinomialDistribution;
import weka.dl4j.distribution.ConstantDistribution;
import weka.dl4j.distribution.Distribution;
import weka.dl4j.distribution.LogNormalDistribution;
import weka.dl4j.distribution.NormalDistribution;
import weka.dl4j.distribution.OrthogonalDistribution;
import weka.dl4j.distribution.TruncatedNormalDistribution;
import weka.dl4j.distribution.UniformDistribution;

public class WeightNoiseTest extends ApiWrapperTest<WeightNoise> {

  @Test
  public void setDistribution() {
    for (Distribution dist :
        new Distribution[]{
            new ConstantDistribution(),
            new LogNormalDistribution(),
            new OrthogonalDistribution(),
            new TruncatedNormalDistribution(),
            new BinomialDistribution(),
            new NormalDistribution(),
            new UniformDistribution()
        }) {
      wrapper.setDistribution(dist);

      assertEquals(dist, wrapper.getDistribution());
    }
  }

  @Test
  public void setApplyToBias() {
    wrapper.setApplyToBias(true);
    assertTrue(wrapper.isApplyToBias());
    wrapper.setApplyToBias(false);
    assertFalse(wrapper.isApplyToBias());
  }

  @Test
  public void setAdditive() {
    wrapper.setAdditive(true);
    assertTrue(wrapper.isAdditive());
    wrapper.setAdditive(false);
    assertFalse(wrapper.isAdditive());
  }

  @Override
  public WeightNoise getApiWrapper() {
    return new WeightNoise();
  }
}
