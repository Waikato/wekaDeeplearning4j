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
public class LSTMTest extends AbstractFeedForwardLayerTest<LSTM>{


  @Override
  public LSTM getApiWrapper(){
    return new  LSTM();
  }

  @Test
  public void testGateActivationFunction(){
    Activation[] acts =
        new Activation[] {
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
