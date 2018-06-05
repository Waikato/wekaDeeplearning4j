package weka.dl4j.layers;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

/**
 * A dense layer test.
 *
 * @author Steven Lang
 */
public class ConvolutionLayerTest extends AbstractFeedForwardLayerTest<ConvolutionLayer> {

  @Before
  @Override
  public void initialize() {
    layer = new ConvolutionLayer();
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
  public void testCudnnAlgoMode() {
    for (AlgoMode mode : AlgoMode.values()) {
      initialize();
      layer.setCudnnAlgoMode(mode);

      assertEquals(mode, layer.getCudnnAlgoMode());
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

}
