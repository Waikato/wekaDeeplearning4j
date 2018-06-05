package weka.dl4j.layers;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.junit.Before;
import org.junit.Test;

/**
 * A subsampling layer test.
 *
 * @author Steven Lang
 */
public class SubsamplingLayerTest extends AbstractLayerTest<SubsamplingLayer> {

  @Before
  @Override
  public void initialize() {
    layer = new SubsamplingLayer();
  }

  @Test
  public void testConvolutionMode() {
    for (ConvolutionMode mode : ConvolutionMode.values()) {
      initialize();
      layer.setConvolutionMode(mode);

      assertEquals(mode, layer.getConvolutionMode());
    }
  }

  @Test
  public void testPoolingType() {
    for (PoolingType type : PoolingType.values()){
      layer.setPoolingType(type);

      assertEquals(type, layer.getPoolingType());
    }
  }

  @Test
  public void testKernelSize(){
    int[] size = {20,20};
    layer.setKernelSize(size);

    assertArrayEquals(size, layer.getKernelSize());
  }

  @Test
  public void testStrideSize(){
    int[] size = {20,20};
    layer.setStride(size);

    assertArrayEquals(size, layer.getStride());

  }
  @Test
  public void testPaddingSize(){
    int[] size = {20,20};
    layer.setPadding(size);

    assertArrayEquals(size, layer.getPadding());

  }

  @Test
  public void testPnorm(){
    int p = 123;
    layer.setPnorm(p);

    assertEquals(p, layer.getPnorm());
  }


  @Test
  public void testEps() {
    double eps = 123.456;
    layer.setEps(eps);

    assertEquals(eps, layer.getEps(), PRECISION);
  }

}
