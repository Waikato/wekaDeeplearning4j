
package weka.dl4j.layers;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * A batch norm layer test.
 *
 * @author Steven Lang
 */
public class BatchNormalizationTest extends AbstractFeedForwardLayerTest<BatchNormalization> {


  @Override
  public BatchNormalization getApiWrapper() {
    return new BatchNormalization();
  }

  @Test
  public void testDecay() {
    double decay = 123.456;
    wrapper.setDecay(decay);

    assertEquals(decay, wrapper.getDecay(), PRECISION);
  }

  @Test
  public void testEps() {
    double eps = 123.456;
    wrapper.setEps(eps);

    assertEquals(eps, wrapper.getEps(), PRECISION);
  }

  @Test
  public void testGamma() {
    double gamma = 123.456;
    wrapper.setGamma(gamma);

    assertEquals(gamma, wrapper.getGamma(), PRECISION);
  }

  @Test
  public void testBeta() {
    double beta = 123.456;
    wrapper.setBeta(beta);

    assertEquals(beta, wrapper.getBeta(), PRECISION);
  }

  @Test
  public void testLockGammaAndBeta() {
    wrapper.setLockGammaAndBeta(true);
    assertTrue(wrapper.getLockGammaAndBeta());
    wrapper.setLockGammaAndBeta(false);
    assertFalse(wrapper.getLockGammaAndBeta());
  }

}
