
package weka.dl4j.layers;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;
import weka.dl4j.lossfunctions.LossBinaryXENT;

/**
 * An output layer test.
 *
 * @author Steven Lang
 */
public class RnnOutputLayerTest extends AbstractFeedForwardLayerTest<RnnOutputLayer>{

  @Override
  public RnnOutputLayer getApiWrapper(){
    return new  RnnOutputLayer();
  }

  @Test
  public void testLossFunction(){
    LossBinaryXENT loss = new LossBinaryXENT();
    wrapper.setLossFn(loss);

    assertEquals(loss, wrapper.getLossFn());
  }
}
