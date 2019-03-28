
package weka.dl4j.schedules;

import java.util.Enumeration;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import weka.core.Option;
import weka.core.OptionMetadata;

/**
 * Exponential schedule for learning rates.
 *
 * @author Steven Lang
 */
@EqualsAndHashCode(callSuper = true)
@ToString
public class ExponentialSchedule extends Schedule<org.nd4j.linalg.schedule.ExponentialSchedule> {

  private static final long serialVersionUID = 6445678599059083075L;

  private double gamma = 0.99;

  @OptionMetadata(
      displayName = "gamma",
      description = "The gamma value (default = 0.99).",
      commandLineParamName = "gamma",
      commandLineParamSynopsis = "-gamma <double>",
      displayOrder = 2
  )
  public double getGamma() {
    return gamma;
  }

  public void setGamma(double gamma) {
    this.gamma = gamma;
  }

  @Override
  public void setBackend(org.nd4j.linalg.schedule.ExponentialSchedule newBackend) {
    this.gamma = newBackend.getGamma();
    this.initialValue = newBackend.getInitialValue();
    this.scheduleType = ScheduleType.fromBackend(newBackend.getScheduleType());
  }

  @Override
  public void initializeBackend() {
    backend = new org.nd4j.linalg.schedule.ExponentialSchedule(scheduleType.getBackend(),
        initialValue, gamma);
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  @Override
  public Enumeration<Option> listOptions() {
    return Option.listOptionsForClassHierarchy(this.getClass(), super.getClass()).elements();
  }

  /**
   * Gets the current settings of the Classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  @Override
  public String[] getOptions() {
    return Option.getOptionsForHierarchy(this, super.getClass());
  }

  /**
   * Parses a given list of options.
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    Option.setOptionsForHierarchy(options, this, super.getClass());
  }
}
