package weka.dl4j.layers;

import org.junit.Before;
import org.junit.Test;

/**
 * An abstract test class for layers.
 *
 * @param <T> Implementing layer class
 * @author Steven Lang
 */
public abstract class AbstractLayerTest<T extends Layer> {

  /**
   * The layer object to be tested
   */
  protected T layer;

  /**
   * Double comparision precision
   */
  protected double PRECISION = 1e-7;

  /**
   * Initialize the layer object.
   */
  @Before
  abstract protected void initialize();
}
