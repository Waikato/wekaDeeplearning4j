package weka.dl4j.layers;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import weka.dl4j.lossfunctions.LossBinaryXENT;
import weka.dl4j.lossfunctions.LossFunction;
/**
 * An output layer test.
 *
 * @author Steven Lang
 */
public class OutputLayerTest extends AbstractFeedForwardLayerTest<OutputLayer>{

  @Before
  @Override
  public void initialize(){
    layer = new OutputLayer();
  }

  @Test
  public void testLossFunction(){
    LossBinaryXENT loss = new LossBinaryXENT();
    layer.setLossFn(loss);

    assertEquals(loss, layer.getLossFn());
  }
}
