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
 * WeightNoiseTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j.weightnoise;

import static org.junit.Assert.*;

import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Assert;
import org.junit.Test;
import weka.dl4j.ApiWrapperTest;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.distribution.BinomialDistribution;
import weka.dl4j.distribution.ConstantDistribution;
import weka.dl4j.distribution.Distribution;
import weka.dl4j.distribution.LogNormalDistribution;
import weka.dl4j.distribution.NormalDistribution;
import weka.dl4j.distribution.OrthogonalDistribution;
import weka.dl4j.distribution.TruncatedNormalDistribution;
import weka.dl4j.distribution.UniformDistribution;

public class WeightNoiseTest extends ApiWrapperTest<WeightNoise> {

  @Test
  public void setDistribution() {
    for (Distribution dist :
        new Distribution[] {
            new ConstantDistribution(),
            new LogNormalDistribution(),
            new OrthogonalDistribution(),
            new TruncatedNormalDistribution(),
            new BinomialDistribution(),
            new NormalDistribution(),
            new UniformDistribution()
        }) {
      wrapper.setDistribution(dist);

      assertEquals(dist, wrapper.getDistribution());
    }
  }

  @Test
  public void setApplyToBias() {
    wrapper.setApplyToBias(true);
    assertTrue(wrapper.isApplyToBias());
    wrapper.setApplyToBias(false);
    assertFalse(wrapper.isApplyToBias());
  }

  @Test
  public void setAdditive() {
    wrapper.setAdditive(true);
    assertTrue(wrapper.isAdditive());
    wrapper.setAdditive(false);
    assertFalse(wrapper.isAdditive());
  }

  @Override
  public WeightNoise getApiWrapper() {
    return new WeightNoise();
  }
}