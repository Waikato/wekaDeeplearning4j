package weka.dl4j.layers;

import static org.junit.Assert.assertEquals;

import org.junit.Before;

/**
 * A dense layer test.
 *
 * @author Steven Lang
 */
public class DenseLayerTest extends AbstractFeedForwardLayerTest<DenseLayer>{


  @Override
  public DenseLayer getApiWrapper(){
    return new  DenseLayer();
  }

}
