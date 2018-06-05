package weka.dl4j.layers;

import static org.junit.Assert.*;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import weka.dl4j.activations.Activation;
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
    layer.setActivationFunction(relu);

    assertEquals(relu, layer.getActivationFunction());
  }

  @Test
  public void testSetNumOut(){
    int nOut = 123;
    layer.setNOut(nOut);

    assertEquals(nOut, layer.getNOut());
  }
}
