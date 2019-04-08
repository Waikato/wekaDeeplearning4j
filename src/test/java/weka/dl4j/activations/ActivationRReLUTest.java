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
 * ActivationRReLUTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j.activations;

import static org.junit.Assert.*;

import org.junit.Test;
import weka.dl4j.ApiWrapperTest;

public class ActivationRReLUTest extends ApiWrapperTest<ActivationRReLU> {

    @Test
    public void setLowerBound() {
        double bound = 123.456;
        wrapper.setLowerBound(bound);

        assertEquals(bound, wrapper.getLowerBound(), PRECISION);
    }

    @Test
    public void setUpperBound() {
        double bound = 123.456;
        wrapper.setUpperBound(bound);

        assertEquals(bound, wrapper.getUpperBound(), PRECISION);
    }

    @Override
    public ActivationRReLU getApiWrapper() {
        return new ActivationRReLU();
    }
}