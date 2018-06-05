package weka.dl4j.layers;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.junit.Before;
import org.junit.Test;
import weka.dl4j.dropout.AbstractDropout;
import weka.dl4j.dropout.AlphaDropout;
import weka.dl4j.dropout.Dropout;
import weka.dl4j.dropout.GaussianDropout;
import weka.dl4j.dropout.GaussianNoise;

/**
 * A global pooling layer test.
 *
 * @author Steven Lang
 */
public class GlobalPoolingLayerTest extends AbstractLayerTest<GlobalPoolingLayer> {

  @Before
  @Override
  public void initialize() {
    layer = new GlobalPoolingLayer();
  }

  @Test
  public void testPoolingType() {
    for (PoolingType type : PoolingType.values()){
      layer.setPoolingType(type);

      assertEquals(type, layer.getPoolingType());
    }
  }

  @Test
  public void testPoolingDimension(){
    int[] dim = {25,25};
    layer.setPoolingDimensions(dim);

    assertArrayEquals(dim, layer.getPoolingDimensions());
  }
  @Test
  public void testCollapseDimensions(){
    layer.setCollapseDimensions(true);
    assertTrue(layer.isCollapseDimensions());
    layer.setCollapseDimensions(false);
    assertFalse(layer.isCollapseDimensions());
  }

  @Test
  public void testPnorm(){
    int p = 123;
    layer.setPnorm(p);

    assertEquals(p, layer.getPnorm());
  }
}
