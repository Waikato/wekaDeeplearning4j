
package weka.dl4j.zoo;

import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import weka.dl4j.Preferences;

/**
 * A WEKA version of DeepLearning4j's InceptionResNetV1 ZooModel.
 *
 * @author Steven Lang
 */
public class InceptionResNetV1 implements ZooModel {

  private static final long serialVersionUID = -520668505548861661L;

  @Override
  public ComputationGraph init(int numLabels, long seed, int[] shape) {
    org.deeplearning4j.zoo.model.InceptionResNetV1 net = org.deeplearning4j.zoo.model.InceptionResNetV1
        .builder()
        .cacheMode(CacheMode.NONE)
        .workspaceMode(Preferences.WORKSPACE_MODE)
        .inputShape(shape)
        .numClasses(numLabels)
        .build();
    return net.init();
  }

  @Override
  public int[][] getShape() {
    return org.deeplearning4j.zoo.model.InceptionResNetV1.builder().build().metaData()
        .getInputShape();
  }
}
