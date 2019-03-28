
package weka.dl4j.layers;

import java.io.Serializable;
import java.util.Enumeration;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.dl4j.ApiWrapper;
import weka.dl4j.ApiWrapperUtil;

/**
 * Abstract layer class.
 * @param <T> Layer implementation
 *
 * @author Steven Lang
 */
@EqualsAndHashCode
@ToString
public abstract class Layer<T extends org.deeplearning4j.nn.conf.layers.Layer>
    implements ApiWrapper<T>, OptionHandler, Serializable {

  private static final long serialVersionUID = 2802598550880773277L;
  T backend;

  public Layer() {
    initializeBackend();
  }

  /**
   * Create an API wrapped layer from a given layer object.
   *
   * @param newBackend Backend object
   * @return API wrapped object
   */
  public static Layer<? extends org.deeplearning4j.nn.conf.layers.Layer> create(
      org.deeplearning4j.nn.conf.layers.Layer newBackend) {
    return ApiWrapperUtil.getImplementingWrapper(Layer.class, newBackend, "weka.dl4j.layers");
  }

  @Override
  public T getBackend() {
    return backend;
  }

  @Override
  public void setBackend(T newBackend) {
    this.backend = newBackend;
  }

  @OptionMetadata(
    displayName = "layer name",
    description = "The name of the layer (default = Batch normalization Layer).",
    commandLineParamName = "name",
    commandLineParamSynopsis = "-name <string>",
    displayOrder = 0
  )
  public String getLayerName() {
    return backend.getLayerName();
  }

  public void setLayerName(String layerName) {
    backend.setLayerName(layerName);
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  @Override
  public Enumeration<Option> listOptions() {

    return Option.listOptionsForClass(this.getClass()).elements();
  }

  /**
   * Gets the current settings of the Classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  @Override
  public String[] getOptions() {

    return Option.getOptions(this, this.getClass());
  }

  /**
   * Parses a given list of options.
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {

    Option.setOptions(options, this, this.getClass());
  }
}
