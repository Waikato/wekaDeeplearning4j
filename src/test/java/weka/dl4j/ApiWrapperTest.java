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
 * ApiWrapperTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j;

import java.util.Arrays;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import weka.core.OptionHandler;

/**
 * An abstract test class for api wrapper objects.
 *
 * @param <T> Implementing api wrapper class
 * @author Steven Lang
 */
public abstract class ApiWrapperTest<T extends ApiWrapper> {

  protected T wrapper;

  /**
   * Double comparision precision
   */
  protected double PRECISION = 1e-7;

  public abstract T getApiWrapper();

  @Before
  public void setup() {
    wrapper = getApiWrapper();
  }

  @Test
  public void testOptions() throws Exception {
    if (wrapper instanceof OptionHandler) {
      OptionHandler optionHandler = ((OptionHandler) wrapper);
      String[] options = optionHandler.getOptions();
      String[] optionsCopy = Arrays.copyOf(options, options.length);
      optionHandler.setOptions(optionsCopy);
      String[] optionsAfter = optionHandler.getOptions();

      Assert.assertArrayEquals(options, optionsAfter);
    }
  }

  @Test
  public void testSettingBackends() {
    T apiWrapper1 = getApiWrapper();
    T apiWrapper2 = getApiWrapper();

    apiWrapper2.setBackend(apiWrapper1.getBackend());

    Assert.assertEquals(apiWrapper1, apiWrapper2);
    Assert.assertEquals(apiWrapper1.getBackend(), apiWrapper2.getBackend());
  }

}
