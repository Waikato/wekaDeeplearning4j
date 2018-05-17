package weka.dl4j.schedules;

import java.util.Enumeration;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import weka.core.Option;
import weka.core.OptionMetadata;

/**
 * Step schedule for learning rates.
 *
 * @author Steven Lang
 */
@EqualsAndHashCode(callSuper = true)
@ToString
public class StepSchedule extends Schedule<org.nd4j.linalg.schedule.StepSchedule> {

  private static final long serialVersionUID = 3282040498224686346L;
  private double step = 10;
  private double decayRate = 0.99;

  @OptionMetadata(
    displayName = "decayRate",
    description = "The decayRate value (default = 0.99).",
    commandLineParamName = "decayRate",
    commandLineParamSynopsis = "-decayRate <double>",
    displayOrder = 2
  )
  public double getDecayRate() {
    return decayRate;
  }

  public void setDecayRate(double decayRate) {
    this.decayRate = decayRate;
  }

  @OptionMetadata(
    displayName = "step",
    description = "The step value (default = 10).",
    commandLineParamName = "step",
    commandLineParamSynopsis = "-step <step>",
    displayOrder = 2
  )
  public double getStep() {
    return step;
  }

  public void setStep(double step) {
    this.step = step;
  }


  @Override
  public void setBackend(org.nd4j.linalg.schedule.StepSchedule newBackend) {
    this.step = backend.getStep();
    this.decayRate = backend.getDecayRate();
    this.initialValue = backend.getInitialValue();
    this.scheduleType = backend.getScheduleType();
  }

  @Override
  public void initializeBackend() {
    new org.nd4j.linalg.schedule.StepSchedule(
        scheduleType, initialValue, decayRate, step);
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
