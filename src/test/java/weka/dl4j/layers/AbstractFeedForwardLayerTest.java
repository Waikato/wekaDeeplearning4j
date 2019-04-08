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
 * AbstractFeedForwardLayerTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j.layers;

import static org.junit.Assert.*;

import org.junit.Test;
import weka.dl4j.activations.ActivationReLU;

/**
 * An abstract test class for feed forward layers.
 *
 * @param <T> Implementing layer class
 * @author Steven Lang
 */
public abstract class AbstractFeedForwardLayerTest<T extends FeedForwardLayer> extends AbstractLayerTest<T> {

    @Test
    public void testActivation() {
        ActivationReLU relu = new ActivationReLU();
        wrapper.setActivationFunction(relu);

        assertEquals(relu, wrapper.getActivationFunction());
    }

    @Test
    public void testSetNumOut() {
        int nOut = 123;
        wrapper.setNOut(nOut);

        assertEquals(nOut, wrapper.getNOut());
    }
}
