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
 * SubsamplingLayerTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j.layers;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.junit.Test;
import weka.dl4j.ConvolutionMode;
import weka.dl4j.PoolingType;

/**
 * A subsampling layer test.
 *
 * @author Steven Lang
 */
public class SubsamplingLayerTest extends AbstractLayerTest<SubsamplingLayer> {

  @Override
  public SubsamplingLayer getApiWrapper() {
    return new SubsamplingLayer();
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
    for (PoolingType type : PoolingType.values()) {
      if (type == PoolingType.NONE)
        continue;

      wrapper.setPoolingType(type);

      assertEquals(type, wrapper.getPoolingType());
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

  @Test
  public void testPnorm() {
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
