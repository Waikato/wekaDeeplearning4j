
package weka.dl4j.activations;

import java.io.Serializable;
import java.util.Enumeration;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import org.nd4j.linalg.activations.IActivation;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.dl4j.ApiWrapper;
import weka.dl4j.ApiWrapperUtil;

/**
 * Abstract activation class
 * @param <T> Activation implementation
 *
 * @author Steven Lang
 */
@EqualsAndHashCode
@ToString
public abstract class Activation<T extends IActivation>
    implements ApiWrapper<T>, OptionHandler, Serializable {

  private static final long serialVersionUID = 2617232662215304733L;

  T backend;

  @Override
  public void setBackend(T newBackend) {
    backend = newBackend;
  }

  @Override
  public T getBackend() {
    return backend;
  }

  public Activation() {
    initializeBackend();
  }

  /**
   * Create an API wrapped schedule from a given ISchedule object.
   *
   * @param newBackend Backend object
   * @return API wrapped object
   */
  public static Activation<? extends IActivation> create(IActivation newBackend) {
    return (Activation<? extends IActivation>) ApiWrapperUtil.getImplementingWrapper(Activation.class, newBackend, "weka.dl4j.activations");
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
