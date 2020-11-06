/*
 * WekaDeeplearning4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * WekaDeeplearning4j is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with WekaDeeplearning4j.  If not, see <https://www.gnu.org/licenses/>.
 *
 * ConvolutionLayerTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j.layers;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.junit.Test;
import weka.dl4j.enums.AlgoMode;
import weka.dl4j.enums.ConvolutionMode;

/**
 * A dense layer test.
 *
 * @author Steven Lang
 */
public class ConvolutionLayerTest extends AbstractFeedForwardLayerTest<ConvolutionLayer> {


  @Override
  public ConvolutionLayer getApiWrapper() {
    return new ConvolutionLayer();
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
  public void testKernelSize() {
    int[] size = {20, 20};
    wrapper.setKernelSize(size);

    assertArrayEquals(size, wrapper.getKernelSize());
  }

  @Test
  public void testStrideSize() {
    int[] size = {20, 20};
    wrapper.setStride(size);

    assertArrayEquals(size, wrapper.getStride());

  }

  @Test
  public void testPaddingSize() {
    int[] size = {20, 20};
    wrapper.setPadding(size);

    assertArrayEquals(size, wrapper.getPadding());

  }

}
