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
 * DropoutLayerTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j.layers;

import static org.junit.Assert.assertEquals;

import lombok.extern.log4j.Log4j2;
import org.junit.Test;
import weka.dl4j.dropout.AbstractDropout;
import weka.dl4j.dropout.AlphaDropout;
import weka.dl4j.dropout.Dropout;
import weka.dl4j.dropout.GaussianDropout;
import weka.dl4j.dropout.GaussianNoise;

/**
 * A dropout layer test.
 *
 * @author Steven Lang
 */
@Log4j2
public class DropoutLayerTest extends AbstractFeedForwardLayerTest<DropoutLayer> {


  @Override
  public DropoutLayer getApiWrapper() {
    return new  DropoutLayer();
  }

  @Test
  public void testDropout() {
    for (AbstractDropout dropout :
        new AbstractDropout[]{
            new AlphaDropout(),
            new Dropout(),
            new GaussianDropout(),
            new GaussianNoise()
        }) {
      wrapper.setDropout(dropout);

      assertEquals(dropout, wrapper.getDropout());
    }
  }

}
