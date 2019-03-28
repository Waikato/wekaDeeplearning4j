
package weka.dl4j.zoo;

import org.deeplearning4j.nn.graph.ComputationGraph;

/**
 * A dummy ZooModel which is empty.
 *
 * @author Steven Lang
 */
public class CustomNet implements ZooModel {

  private static final long serialVersionUID = 7131900848379752732L;

  @Override
  public ComputationGraph init(int numLabels, long seed, int[] shape)
      throws UnsupportedOperationException {
    throw new UnsupportedOperationException(
        "This model cannot be initialized as a MultiLayerNetwork.");
  }

  @Override
  public int[][] getShape() {
    return new int[0][0];
  }
}
