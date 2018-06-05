package weka.dl4j.layers;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

/**
 * A batch norm layer test.
 *
 * @author Steven Lang
 */
public class BatchNormalizationTest extends AbstractFeedForwardLayerTest<BatchNormalization> {

  @Before
  @Override
  public void initialize() {
    layer = new BatchNormalization();
  }

  @Test
  public void testDecay() {
    double decay = 123.456;
    layer.setDecay(decay);

    assertEquals(decay, layer.getDecay(), PRECISION);
  }

  @Test
  public void testEps() {
    double eps = 123.456;
    layer.setEps(eps);

    assertEquals(eps, layer.getEps(), PRECISION);
  }

  @Test
  public void testGamma() {
    double gamma = 123.456;
    layer.setGamma(gamma);

    assertEquals(gamma, layer.getGamma(), PRECISION);
  }

  @Test
  public void testBeta() {
    double beta = 123.456;
    layer.setBeta(beta);

    assertEquals(beta, layer.getBeta(), PRECISION);
  }

  @Test
  public void testLockGammaAndBeta() {
    layer.setLockGammaAndBeta(true);
    assertTrue(layer.getLockGammaAndBeta());
    layer.setLockGammaAndBeta(false);
    assertFalse(layer.getLockGammaAndBeta());
  }

}
