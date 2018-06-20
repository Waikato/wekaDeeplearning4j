package weka.dl4j.layers;

import static org.junit.Assert.assertEquals;

import lombok.extern.log4j.Log4j2;
import org.junit.Test;
import weka.dl4j.dropout.AbstractDropout;
import weka.dl4j.dropout.AlphaDropout;
import weka.dl4j.dropout.Dropout;
import weka.dl4j.dropout.GaussianDropout;
import weka.dl4j.dropout.GaussianNoise;

/**
 * A dropout layer test.
 *
 * @author Steven Lang
 */
@Log4j2
public class DropoutLayerTest extends AbstractFeedForwardLayerTest<DropoutLayer> {


  @Override
  public DropoutLayer getApiWrapper() {
    return new  DropoutLayer();
  }

  @Test
  public void testDropout() {
    for (AbstractDropout dropout :
        new AbstractDropout[]{
            new AlphaDropout(),
            new Dropout(),
            new GaussianDropout(),
            new GaussianNoise()
        }) {
      wrapper.setDropout(dropout);

      assertEquals(dropout, wrapper.getDropout());
    }
  }

}
