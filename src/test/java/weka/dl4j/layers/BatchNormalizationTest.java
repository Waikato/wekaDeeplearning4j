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
 * BatchNormalizationTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j.layers;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Before;
import org.junit.Test;

/**
 * A batch norm layer test.
 *
 * @author Steven Lang
 */
public class BatchNormalizationTest extends AbstractFeedForwardLayerTest<BatchNormalization> {


  @Override
  public BatchNormalization getApiWrapper() {
    return new  BatchNormalization();
  }

  @Test
  public void testDecay() {
    double decay = 123.456;
    wrapper.setDecay(decay);

    assertEquals(decay, wrapper.getDecay(), PRECISION);
  }

  @Test
  public void testEps() {
    double eps = 123.456;
    wrapper.setEps(eps);

    assertEquals(eps, wrapper.getEps(), PRECISION);
  }

  @Test
  public void testGamma() {
    double gamma = 123.456;
    wrapper.setGamma(gamma);

    assertEquals(gamma, wrapper.getGamma(), PRECISION);
  }

  @Test
  public void testBeta() {
    double beta = 123.456;
    wrapper.setBeta(beta);

    assertEquals(beta, wrapper.getBeta(), PRECISION);
  }

  @Test
  public void testLockGammaAndBeta() {
    wrapper.setLockGammaAndBeta(true);
    assertTrue(wrapper.getLockGammaAndBeta());
    wrapper.setLockGammaAndBeta(false);
    assertFalse(wrapper.getLockGammaAndBeta());
  }

}
