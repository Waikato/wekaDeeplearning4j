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
 * LSTMTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j.layers;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;
import weka.dl4j.activations.Activation;
import weka.dl4j.activations.ActivationCube;
import weka.dl4j.activations.ActivationELU;
import weka.dl4j.activations.ActivationHardSigmoid;
import weka.dl4j.activations.ActivationHardTanH;
import weka.dl4j.activations.ActivationIdentity;
import weka.dl4j.activations.ActivationLReLU;
import weka.dl4j.activations.ActivationRReLU;
import weka.dl4j.activations.ActivationRationalTanh;
import weka.dl4j.activations.ActivationReLU;
import weka.dl4j.activations.ActivationSoftPlus;
import weka.dl4j.activations.ActivationSoftSign;
import weka.dl4j.activations.ActivationSoftmax;

/**
 * A LSTM layer test.
 *
 * @author Steven Lang
 */
public class LSTMTest extends AbstractFeedForwardLayerTest<LSTM> {


    @Override
    public LSTM getApiWrapper() {
        return new LSTM();
    }

    @Test
    public void testGateActivationFunction() {
        Activation[] acts =
                new Activation[]{
                        new ActivationCube(),
                        new ActivationELU(),
                        new ActivationHardSigmoid(),
                        new ActivationHardTanH(),
                        new ActivationIdentity(),
                        new ActivationLReLU(),
                        new ActivationRationalTanh(),
                        new ActivationReLU(),
                        new ActivationRReLU(),
                        new ActivationHardSigmoid(),
                        new ActivationSoftmax(),
                        new ActivationSoftPlus(),
                        new ActivationSoftSign(),
                        new ActivationHardTanH()
                };
        for (Activation act : acts) {
            wrapper.setGateActivationFn(act);

            assertEquals(act, wrapper.getGateActivationFn());
        }
    }
}
