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
 * AdamTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j.updater;

import static org.junit.Assert.*;

import org.junit.Test;

public class AdamTest extends AbstractUpdaterTest<Adam> {

  @Test
  public void setBeta1() {
    double value = 123.456;
    wrapper.setBeta1(value);

    assertEquals(value, wrapper.getBeta1(), PRECISION);
  }

  @Test
  public void setBeta2() {
    double value = 123.456;
    wrapper.setBeta2(value);

    assertEquals(value, wrapper.getBeta2(), PRECISION);
  }

  @Test
  public void setEpsilon() {
    double value = 123.456;
    wrapper.setEpsilon(value);

    assertEquals(value, wrapper.getEpsilon(), PRECISION);
  }

  @Override
  public Adam getApiWrapper() {
    return new Adam();
  }
}