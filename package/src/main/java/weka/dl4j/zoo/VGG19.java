package weka.dl4j.zoo;

import org.deeplearning4j.nn.graph.ComputationGraph;

/**
 * A WEKA version of DeepLearning4j's VGGN19 ZooModel.
 *
 * @author Steven Lang
 */
public class VGG19 implements ZooModel {
  private static final long serialVersionUID = -4452023767749633607L;

  @Override
  public ComputationGraph init(int numLabels, long seed, int[][] shape) {
    org.deeplearning4j.zoo.model.VGG19 net =
        new org.deeplearning4j.zoo.model.VGG19(numLabels, seed, 1);
    net.setInputShape(shape);
    org.deeplearning4j.nn.conf.MultiLayerConfiguration conf = net.conf();
    return mlpToCG(conf, shape);
  }

  @Override
  public int[][] getShape() {
    return new org.deeplearning4j.zoo.model.VGG19().metaData().getInputShape();
  }
}
