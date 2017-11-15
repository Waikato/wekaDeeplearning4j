package weka.dl4j.zoo;

import org.deeplearning4j.nn.graph.ComputationGraph;

/**
 * A WEKA version of DeepLearning4j's LeNet ZooModel.
 *
 * @author Steven Lang
 */
public class LeNet implements ZooModel {
  private static final long serialVersionUID = 7790142171346455139L;

  @Override
  public ComputationGraph init(int numLabels, long seed, int[][] shape) {
    org.deeplearning4j.zoo.model.LeNet net =
        new org.deeplearning4j.zoo.model.LeNet(numLabels, seed, 1);
    net.setInputShape(shape);
    org.deeplearning4j.nn.conf.MultiLayerConfiguration conf = net.conf();
    return mlpToCG(conf, shape);
  }

  @Override
  public int[][] getShape() {
    return new org.deeplearning4j.zoo.model.LeNet().metaData().getInputShape();
  }
}
