package weka.dl4j.schedules;

import java.util.Enumeration;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import weka.core.Option;
import weka.core.OptionMetadata;

/**
 * Inverse exponential schedule for learning rates.
 *
 * @author Steven Lang
 */
@EqualsAndHashCode(callSuper = true)
@ToString
public class InverseSchedule extends Schedule<org.nd4j.linalg.schedule.InverseSchedule> {

  private static final long serialVersionUID = 1840128418738133390L;
  private double gamma = 0.99;
  private double power = 1;

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

  @OptionMetadata(
    displayName = "power",
    description = "The power value (default = 1.0).",
    commandLineParamName = "power",
    commandLineParamSynopsis = "-power <double>",
    displayOrder = 3
  )
  public double getPower() {
    return power;
  }

  public void setPower(double power) {
    this.power = power;
  }

  @Override
  public org.nd4j.linalg.schedule.InverseSchedule getBackend() {
    return new org.nd4j.linalg.schedule.InverseSchedule(scheduleType, initialValue, gamma, power);
  }

  @Override
  public void initializeBackend() {
    backend =
        new org.nd4j.linalg.schedule.InverseSchedule(scheduleType, initialValue, gamma, power);
  }

  @Override
  public void setBackend(org.nd4j.linalg.schedule.InverseSchedule newBackend) {
    this.gamma = newBackend.getGamma();
    this.power = newBackend.getPower();
    this.initialValue = newBackend.getInitialValue();
    this.scheduleType = newBackend.getScheduleType();
  }

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  @Override
  public Enumeration<Option> listOptions() {
    return Option.listOptionsForClassHierarchy(this.getClass(),super.getClass()).elements();
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
