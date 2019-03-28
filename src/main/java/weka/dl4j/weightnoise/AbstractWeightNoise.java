
package weka.dl4j.weightnoise;

import java.io.Serializable;
import java.util.Enumeration;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import org.deeplearning4j.nn.conf.weightnoise.IWeightNoise;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.dl4j.ApiWrapper;
import weka.dl4j.ApiWrapperUtil;

/**
 * Abstract weight noise class.
 *
 * @param <T> Weight noise implementation
 * @author Steven Lang
 */
@EqualsAndHashCode
@ToString
public abstract class AbstractWeightNoise<T extends IWeightNoise> implements ApiWrapper<T>,
    OptionHandler, Serializable {

  private static final long serialVersionUID = 910666004504402198L;
  T backend;

  public AbstractWeightNoise() {
    initializeBackend();
  }

  /**
   * Create an API wrapped updater from a given updater object.
   *
   * @param newBackend Backend object
   * @return API wrapped object
   */
  public static AbstractWeightNoise<? extends IWeightNoise> create(IWeightNoise newBackend) {
    return ApiWrapperUtil
        .getImplementingWrapper(AbstractWeightNoise.class, newBackend, "weka.dl4j.updater");
  }

  @Override
  public T getBackend() {
    return backend;
  }

  @Override
  public void setBackend(T newBackend) {
    backend = newBackend;
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
