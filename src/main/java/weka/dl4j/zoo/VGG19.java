
package weka.dl4j.zoo;

import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import weka.dl4j.Preferences;

/**
 * A WEKA version of DeepLearning4j's VGGN19 ZooModel.
 *
 * @author Steven Lang
 */
public class VGG19 implements ZooModel {

  private static final long serialVersionUID = -4452023767749633607L;

  @Override
  public ComputationGraph init(int numLabels, long seed, int[] shape) {
    org.deeplearning4j.zoo.model.VGG19 net = org.deeplearning4j.zoo.model.VGG19.builder()
        .cacheMode(CacheMode.NONE)
        .workspaceMode(Preferences.WORKSPACE_MODE)
        .inputShape(shape)
        .numClasses(numLabels)
        .build();
    org.deeplearning4j.nn.conf.ComputationGraphConfiguration conf = net.conf();
    return new ComputationGraph(conf);
  }

  @Override
  public int[][] getShape() {
    return org.deeplearning4j.zoo.model.VGG19.builder().build().metaData().getInputShape();
  }
}
