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
 * GlobalPoolingLayerTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j.layers;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;
import weka.dl4j.enums.PoolingType;

/**
 * A global pooling layer test.
 *
 * @author Steven Lang
 */
public class GlobalPoolingLayerTest extends AbstractLayerTest<GlobalPoolingLayer> {


  @Override
  public GlobalPoolingLayer getApiWrapper() {
    return new GlobalPoolingLayer();
  }

  @Test
  public void testPoolingType() {
    for (PoolingType type : PoolingType.values()) {
      if (type.isCustom())
        continue;

      wrapper.setPoolingType(type);

      assertEquals(type, wrapper.getPoolingType());
    }
  }

  @Test
  public void testPoolingDimension() {
    int[] dim = {25, 25};
    wrapper.setPoolingDimensions(dim);

    assertArrayEquals(dim, wrapper.getPoolingDimensions());
  }

  @Test
  public void testCollapseDimensions() {
    wrapper.setCollapseDimensions(true);
    assertTrue(wrapper.isCollapseDimensions());
    wrapper.setCollapseDimensions(false);
    assertFalse(wrapper.isCollapseDimensions());
  }

  @Test
  public void testPnorm() {
    int p = 123;
    wrapper.setPnorm(p);

    assertEquals(p, wrapper.getPnorm());
  }
}
