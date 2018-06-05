package weka.dl4j.schedules;

import java.util.Enumeration;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import weka.core.Option;
import weka.core.OptionMetadata;

/**
 * Sigmoid schedule for learning rates.
 *
 * @author Steven Lang
 */
@EqualsAndHashCode(callSuper = true)
@ToString
public class SigmoidSchedule extends Schedule<org.nd4j.linalg.schedule.SigmoidSchedule> {

  private static final long serialVersionUID = 3282040498224686346L;
  private int stepSize = 10;
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

  @OptionMetadata(
    displayName = "stepSize",
    description = "The stepSize value (default = 10).",
    commandLineParamName = "stepSize",
    commandLineParamSynopsis = "-stepSize <stepSize>",
    displayOrder = 2
  )
  public int getStepSize() {
    return stepSize;
  }

  public void setStepSize(int stepSize) {
    this.stepSize = stepSize;
  }

  @Override
  public void initializeBackend() {
    backend = new org.nd4j.linalg.schedule.SigmoidSchedule(
        scheduleType, initialValue, gamma, stepSize);
  }

  @Override
  public void setBackend(org.nd4j.linalg.schedule.SigmoidSchedule newBackend) {
    this.stepSize = backend.getStepSize();
    this.gamma = backend.getGamma();
    this.initialValue = backend.getInitialValue();
    this.scheduleType = backend.getScheduleType();
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
