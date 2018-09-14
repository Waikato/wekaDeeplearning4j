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
 * MapScheduleTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j.schedules;

import static org.junit.Assert.*;

import com.google.common.collect.ImmutableMap;
import java.util.Collections;
import org.junit.Test;
import weka.dl4j.ApiWrapperTest;

public class MapScheduleTest extends AbstractScheduleTest<MapSchedule> {

  @Test
  public void setValues() {
    ImmutableMap<Integer, Double> values = ImmutableMap.of(0, 0.1, 1, 0.2, 2, 0.3, 3, 0.4);
    wrapper.setValues(values);

    assertEquals(values, wrapper.getValues());
  }

  @Override
  public MapSchedule getApiWrapper() {
    return new MapSchedule();
  }
}