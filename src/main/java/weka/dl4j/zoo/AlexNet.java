package weka.dl4j.zoo;

import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import weka.dl4j.Preferences;

/**
 * A WEKA version of DeepLearning4j's LeNet ZooModel.
 *
 * @author Steven Lang
 */
public class AlexNet implements ZooModel {

  private static final long serialVersionUID = -520668505548861661L;

  @Override
  public ComputationGraph init(int numLabels, long seed, int[][] shape) {
    org.deeplearning4j.zoo.model.AlexNet net =
        new org.deeplearning4j.zoo.model.AlexNet(numLabels, seed, Preferences.WORKSPACE_MODE);
    net.setInputShape(shape);
    org.deeplearning4j.nn.conf.MultiLayerConfiguration conf = net.conf();

    return mlpToCG(conf, shape);
  }

  @Override
  public int[][] getShape() {
    return new org.deeplearning4j.zoo.model.AlexNet().metaData().getInputShape();
  }
}
