
package weka.dl4j.layers;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import weka.dl4j.PoolingType;
import org.junit.Before;
import org.junit.Test;

/**
 * A global pooling layer test.
 *
 * @author Steven Lang
 */
public class GlobalPoolingLayerTest extends AbstractLayerTest<GlobalPoolingLayer> {


  @Override
  public GlobalPoolingLayer getApiWrapper() {
    return new  GlobalPoolingLayer();
  }

  @Test
  public void testPoolingType() {
    for (PoolingType type : PoolingType.values()){
      wrapper.setPoolingType(type);

      assertEquals(type, wrapper.getPoolingType());
    }
  }

  @Test
  public void testPoolingDimension(){
    int[] dim = {25,25};
    wrapper.setPoolingDimensions(dim);

    assertArrayEquals(dim, wrapper.getPoolingDimensions());
  }
  @Test
  public void testCollapseDimensions(){
    wrapper.setCollapseDimensions(true);
    assertTrue(wrapper.isCollapseDimensions());
    wrapper.setCollapseDimensions(false);
    assertFalse(wrapper.isCollapseDimensions());
  }

  @Test
  public void testPnorm(){
    int p = 123;
    wrapper.setPnorm(p);

    assertEquals(p, wrapper.getPnorm());
  }
}
