
package weka.dl4j.dropout;

import java.util.Enumeration;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import org.nd4j.linalg.schedule.ISchedule;
import weka.core.Option;
import weka.core.OptionMetadata;
import weka.dl4j.schedules.ConstantSchedule;
import weka.dl4j.schedules.Schedule;

/**
 * Gaussian dropout implementation.
 *
 * @author Steven Lang
 */
@EqualsAndHashCode(callSuper = true)
@ToString
public class GaussianDropout
    extends AbstractDropout<org.deeplearning4j.nn.conf.dropout.GaussianDropout> {

  private static final long serialVersionUID = -294245467732026881L;
  private Schedule<? extends ISchedule> rateSchedule;

  @OptionMetadata(
      displayName = "schedule",
      description = "The rate schedule (default = ConstantScheduleImpl).",
      commandLineParamName = "schedule",
      commandLineParamSynopsis = "-schedule <Schedule>",
      displayOrder = 2
  )
  public Schedule<? extends ISchedule> getRateSchedule() {
    return rateSchedule;
  }

  public void setRateSchedule(Schedule<? extends ISchedule> rateSchedule) {
    this.rateSchedule = rateSchedule;
  }

  @Override
  public org.deeplearning4j.nn.conf.dropout.GaussianDropout getBackend() {
    return new org.deeplearning4j.nn.conf.dropout.GaussianDropout(rateSchedule.getBackend());
  }

  @Override
  public void initializeBackend() {
    rateSchedule = new ConstantSchedule();
    this.backend = new org.deeplearning4j.nn.conf.dropout.GaussianDropout(
        rateSchedule.getBackend());
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
