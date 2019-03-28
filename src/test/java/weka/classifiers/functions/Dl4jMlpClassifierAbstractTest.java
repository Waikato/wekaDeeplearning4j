
package weka.classifiers.functions;

import junit.framework.Test;
import junit.framework.TestSuite;
import weka.dl4j.layers.Layer;
import org.nd4j.linalg.lossfunctions.impl.LossSquaredHinge;
import weka.classifiers.AbstractClassifierTest;
import weka.classifiers.Classifier;
import weka.dl4j.earlystopping.EarlyStopping;
import weka.dl4j.layers.DenseLayer;
import weka.dl4j.layers.OutputLayer;

/**
 * Abstract classifier test for the {@link Dl4jMlpClassifier}.
 *
 * @author Steven Lang
 */
public class Dl4jMlpClassifierAbstractTest extends AbstractClassifierTest {

  public Dl4jMlpClassifierAbstractTest(String name) {
    super(name);
  }

  public static Test suite() {
    return new TestSuite(Dl4jMlpClassifierAbstractTest.class);
  }

  public static void main(String[] args) throws Exception {
    junit.textui.TestRunner.run(suite());
  }

  @Override
  public Classifier getClassifier() {
    Dl4jMlpClassifier mlp = new Dl4jMlpClassifier();
    DenseLayer dl = new DenseLayer();
    dl.setNOut(2);
    OutputLayer ol = new OutputLayer();
    ol.setLossFn(new weka.dl4j.lossfunctions.LossSquaredHinge());
    mlp.setLayers(dl, ol);
    mlp.setNumEpochs(1);
    mlp.setEarlyStopping(new EarlyStopping(0, 0));
    return mlp;
  }
}
