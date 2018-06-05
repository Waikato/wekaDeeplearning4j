package weka.dl4j.zoo;

import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import weka.dl4j.Preferences;

/**
 * A WEKA version of DeepLearning4j's VGGN16 ZooModel.
 *
 * @author Steven Lang
 */
public class VGG16 implements ZooModel {
  private static final long serialVersionUID = -6728816089752609851L;

  @Override
  public ComputationGraph init(int numLabels, long seed, int[][] shape) {
    org.deeplearning4j.zoo.model.VGG16 net =
        new org.deeplearning4j.zoo.model.VGG16(numLabels, seed,  Preferences.WORKSPACE_MODE);
    net.setInputShape(shape);
    org.deeplearning4j.nn.conf.MultiLayerConfiguration conf = net.conf();
    return mlpToCG(conf, shape);
  }

  @Override
  public int[][] getShape() {
    return new org.deeplearning4j.zoo.model.VGG16().metaData().getInputShape();
  }
}
