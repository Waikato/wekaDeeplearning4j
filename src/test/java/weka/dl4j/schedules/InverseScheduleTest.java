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
 * InverseScheduleTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j.schedules;

import static org.junit.Assert.*;

import org.junit.Test;
import weka.dl4j.ApiWrapperTest;

public class InverseScheduleTest extends AbstractScheduleTest<InverseSchedule> {

  @Test
  public void setGamma() {
    double value = 123.456;
    wrapper.setGamma(value);

    assertEquals(value, wrapper.getGamma(), PRECISION);

  }

  @Test
  public void setPower() {
    double value = 123.456;
    wrapper.setPower(value);

    assertEquals(value, wrapper.getPower(), PRECISION);
  }

  @Override
  public InverseSchedule getApiWrapper() {
    return new InverseSchedule();
  }
}