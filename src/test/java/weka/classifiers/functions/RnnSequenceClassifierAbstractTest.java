
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
import weka.dl4j.layers.RnnOutputLayer;
/**
 * Abstract classifier test for the {@link RnnSequenceClassifier}.
 *
 * @author Steven Lang
 */
public class RnnSequenceClassifierAbstractTest extends AbstractClassifierTest {

  public RnnSequenceClassifierAbstractTest(String name) {
    super(name);
  }

  public static Test suite() {
    return new TestSuite(RnnSequenceClassifierAbstractTest.class);
  }

  public static void main(String[] args) {
    junit.textui.TestRunner.run(suite());
  }

  @Override
  public Classifier getClassifier() {
    RnnSequenceClassifier rnn = new RnnSequenceClassifier();
    RnnOutputLayer ol = new RnnOutputLayer();
    rnn.setLayers(ol);
    rnn.setNumEpochs(1);
    rnn.setEarlyStopping(new EarlyStopping(0, 0));
    return rnn;
  }
}
