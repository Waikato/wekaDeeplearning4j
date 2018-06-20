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
  public void testSetNumOut(){
    int nOut = 123;
    wrapper.setNOut(nOut);

    assertEquals(nOut, wrapper.getNOut());
  }
}
