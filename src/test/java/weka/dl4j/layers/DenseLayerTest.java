
package weka.dl4j.layers;

/**
 * A dense layer test.
 *
 * @author Steven Lang
 */
public class DenseLayerTest extends AbstractFeedForwardLayerTest<DenseLayer> {


  @Override
  public DenseLayer getApiWrapper() {
    return new DenseLayer();
  }

}
