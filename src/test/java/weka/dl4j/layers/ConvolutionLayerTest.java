package weka.dl4j.layers;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import weka.dl4j.ConvolutionMode;
import weka.dl4j.AlgoMode;
import org.junit.Before;
import org.junit.Test;

/**
 * A dense layer test.
 *
 * @author Steven Lang
 */
public class ConvolutionLayerTest extends AbstractFeedForwardLayerTest<ConvolutionLayer> {


  @Override
  public ConvolutionLayer getApiWrapper() {
    return new  ConvolutionLayer();
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
  public void testCudnnAlgoMode() {
    for (AlgoMode mode : AlgoMode.values()) {
      getApiWrapper();
      wrapper.setCudnnAlgoMode(mode);

      assertEquals(mode, wrapper.getCudnnAlgoMode());
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

}
