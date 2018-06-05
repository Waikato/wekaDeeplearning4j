package weka.dl4j.layers;

import weka.core.OptionMetadata;
import weka.dl4j.activations.Activation;

public abstract class FeedForwardLayer<T extends org.deeplearning4j.nn.conf.layers.FeedForwardLayer>
    extends Layer<T> {

  private static final long serialVersionUID = -909125865597886097L;

  public FeedForwardLayer() {
    super();
    backend.setL1(Double.NaN);
    backend.setL2(Double.NaN);
    backend.setGradientNormalizationThreshold(Double.NaN);
    backend.setGradientNormalization(null);
    backend.setBiasInit(Double.NaN);
  }

  @OptionMetadata(
    displayName = "activation function",
    description = "The activation function to use (default = Identity).",
    commandLineParamName = "activation",
    commandLineParamSynopsis = "-activation <specification>",
    displayOrder = 1
  )
  public Activation getActivationFunction() {
    return Activation.create(backend.getActivationFn());
  }

  public void setActivationFunction(Activation activationFn) {
    backend.setActivationFn(activationFn.getBackend());
  }

  @OptionMetadata(
    displayName = "number of outputs",
    description = "The number of outputs.",
    commandLineParamName = "nOut",
    commandLineParamSynopsis = "-nOut <int>",
    displayOrder = 2
  )
  public int getNOut() {
    return backend.getNOut();
  }

  public void setNOut(int nOut) {
    backend.setNOut(nOut);
  }
}
