package weka.dl4j.zoo;

import org.deeplearning4j.nn.graph.ComputationGraph;

/**
 * A WEKA version of DeepLearning4j's ResNet50 ZooModel.
 *
 * @author Steven Lang
 */
public class ResNet50 implements ZooModel {
  private static final long serialVersionUID = -520668505548861661L;

  @Override
  public ComputationGraph init(int numLabels, long seed, int[][] shape) {
    org.deeplearning4j.zoo.model.ResNet50 net =
        new org.deeplearning4j.zoo.model.ResNet50(numLabels, seed, 1);
    net.setInputShape(shape);
    return net.init();
  }

  @Override
  public int[][] getShape() {
    return new org.deeplearning4j.zoo.model.ResNet50().metaData().getInputShape();
  }
}
