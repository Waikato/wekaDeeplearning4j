package weka.dl4j.layers;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import weka.dl4j.lossfunctions.LossBinaryXENT;
import weka.dl4j.lossfunctions.LossFunction;

/**
 * A dense layer test.
 *
 * @author Steven Lang
 */
public class DenseLayerTest extends AbstractFeedForwardLayerTest<DenseLayer>{

  @Before
  @Override
  public void initialize(){
    layer = new DenseLayer();
  }

}
