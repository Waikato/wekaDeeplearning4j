package weka.dl4j.zoo;

import org.deeplearning4j.nn.graph.ComputationGraph;

/**
 * A WEKA version of DeepLearning4j's FaceNetNN4Small2 ZooModel.
 *
 * @author Steven Lang
 */
public class FaceNetNN4Small2 implements ZooModel {
  private static final long serialVersionUID = -520668505548861661L;

  @Override
  public ComputationGraph init(int numLabels, long seed, int[][] shape) {
    org.deeplearning4j.zoo.model.FaceNetNN4Small2 net =
        new org.deeplearning4j.zoo.model.FaceNetNN4Small2(numLabels, seed, 1);
    net.setInputShape(shape);
    return net.init();
  }

  @Override
  public int[][] getShape() {
    return new org.deeplearning4j.zoo.model.FaceNetNN4Small2().metaData().getInputShape();
  }
}
