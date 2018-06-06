package weka.dl4j.layers;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.junit.Before;
import org.junit.Test;

/**
 * A subsampling layer test.
 *
 * @author Steven Lang
 */
public class SubsamplingLayerTest extends AbstractLayerTest<SubsamplingLayer> {

  @Override
  public SubsamplingLayer getApiWrapper() {
    return new  SubsamplingLayer();
  }

  @Test
  public void testConvolutionMode() {
    for (ConvolutionMode mode : ConvolutionMode.values()) {
      getApiWrapper();
      wrapper.setConvolutionMode(mode);

      assertEquals(mode, wrapper.getConvolutionMode());
    }
  }

  @Test
  public void testPoolingType() {
    for (PoolingType type : PoolingType.values()){
      wrapper.setPoolingType(type);

      assertEquals(type, wrapper.getPoolingType());
    }
  }

  @Test
  public void testKernelSize(){
    int[] size = {20,20};
    wrapper.setKernelSize(size);

    assertArrayEquals(size, wrapper.getKernelSize());
  }

  @Test
  public void testStrideSize(){
    int[] size = {20,20};
    wrapper.setStride(size);

    assertArrayEquals(size, wrapper.getStride());

  }
  @Test
  public void testPaddingSize(){
    int[] size = {20,20};
    wrapper.setPadding(size);

    assertArrayEquals(size, wrapper.getPadding());

  }

  @Test
  public void testPnorm(){
    int p = 123;
    wrapper.setPnorm(p);

    assertEquals(p, wrapper.getPnorm());
  }


  @Test
  public void testEps() {
    double eps = 123.456;
    wrapper.setEps(eps);

    assertEquals(eps, wrapper.getEps(), PRECISION);
  }

}
