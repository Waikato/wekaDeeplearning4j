package weka.dl4j.activations;

import org.deeplearning4j.nn.conf.layers.Layer;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.Instances;
import weka.dl4j.layers.DenseLayer;
import weka.dl4j.layers.OutputLayer;
import weka.util.DatasetLoader;

/**
 * JUnit tests for all available activations
 *
 * @author Steven Lang
 */
public class ActivationTest {
  /**
   * Run dummy network with give activationfunction for the first layer
   *
   * @param act Activation function to test
   * @throws Exception Something went wrong.
   */
  public static void runClf(IActivation act) throws Exception {
    Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
    // Data
    DenseLayer denseLayer = new DenseLayer();
    denseLayer.setNOut(2);
    denseLayer.setLayerName("Dense-layer");
    denseLayer.setActivationFn(act);

    OutputLayer outputLayer = new OutputLayer();
    outputLayer.setActivationFn(Activation.SOFTMAX.getActivationFunction());
    outputLayer.setLayerName("Output-layer");

    clf.setNumEpochs(1);
    clf.setLayers(new Layer[] {denseLayer, outputLayer});

    final Instances data = DatasetLoader.loadIris();
    clf.buildClassifier(data);
    clf.distributionsForInstances(data);
  }

  /**
   * Test all activation functions with a dummy network
   *
   * @throws Exception
   */
  @Test
  public void testActivations() throws Exception {
    IActivation[] acts =
        new IActivation[] {
          new ActivationCube(),
          new ActivationELU(),
          new ActivationHardSigmoid(),
          new ActivationHardTanH(),
          new ActivationIdentity(),
          new ActivationLReLU(),
          new ActivationRationalTanh(),
          new ActivationReLU(),
          new ActivationRReLU(),
          new ActivationHardSigmoid(),
          new ActivationSoftmax(),
          new ActivationSoftPlus(),
          new ActivationSoftSign(),
          new ActivationHardTanH()
        };
    for (IActivation act : acts) {
      runClf(act);
    }
  }
}
