package weka.dl4j.distribution;

import java.io.Serializable;
import java.util.Enumeration;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.conf.layers.Layer;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.dl4j.ApiWrapper;
import weka.dl4j.ApiWrapperUtil;

@EqualsAndHashCode
public abstract class Distribution<T extends org.deeplearning4j.nn.conf.distribution.Distribution>
    implements ApiWrapper<T>, OptionHandler, Serializable {

  private static final long serialVersionUID = -283820469693291697L;
  T backend;

  public Distribution() {
    initializeBackend();
  }

  /**
   * Create an API wrapped layer from a given layer object.
   *
   * @param newBackend Backend object
   * @return API wrapped object
   */
  public static Distribution<? extends org.deeplearning4j.nn.conf.distribution.Distribution> create(
      Distribution<? extends org.deeplearning4j.nn.conf.distribution.Distribution> newBackend) {
    return ApiWrapperUtil.getImplementingWrapper(Distribution.class, newBackend, "weka.dl4j.distribution");
  }

  @Override
  public T getBackend() {
    return backend;
  }

  @Override
  public void setBackend(T newBackend) {
    this.backend = newBackend;
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
