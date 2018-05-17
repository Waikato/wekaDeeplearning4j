package weka.dl4j.schedules;

import java.io.Serializable;
import java.util.Enumeration;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.dl4j.ApiWrapper;
import weka.dl4j.ApiWrapperUtil;

/**
 * Default schedule interface that implements WEKA option handling. Each class that implements
 * Schedulers extending from this class must have a backing implementation of the referencing Dl4j
 * scheduler since the scheduler in Dl4j do not allow access to their class members (private final).
 *
 * SigmoidSchedule, StepSchedule)
 *
 * @author Steven Lang
 */
@EqualsAndHashCode
@ToString
public abstract class Schedule<T extends ISchedule>
    implements OptionHandler, ApiWrapper<T>, Serializable {

  private static final long serialVersionUID = 5588471135175058051L;

  /** Schedule type */
  ScheduleType scheduleType = ScheduleType.EPOCH;

  /** Initial value */
  double initialValue = 1.0;

  /** Schedule that is backing the implementation */
  T backend;

  public Schedule() {
    initializeBackend();
  }

  @OptionMetadata(
    displayName = "scheduleType",
    description = "The schedule type, one of {EPOCH,ITERATION} (default = EPOCH).",
    commandLineParamName = "scheduleType",
    commandLineParamSynopsis = "-scheduleType <string>",
    displayOrder = 1
  )
  public ScheduleType getScheduleType() {
    return scheduleType;
  }

  public void setScheduleType(ScheduleType scheduleType) {
    this.scheduleType = scheduleType;
  }

  @OptionMetadata(
    displayName = "initialValue",
    description = "The initial value (default = 1.0).",
    commandLineParamName = "scheduleType",
    commandLineParamSynopsis = "-scheduleType <string>",
    displayOrder = 2
  )
  public double getInitialValue() {
    return initialValue;
  }

  public void setInitialValue(double initialValue) {
    this.initialValue = initialValue;
  }

  /**
   * Create an API wrapped schedule from a given ISchedule object.
   *
   * @param newBackend Backend object
   * @return API wrapped object
   */
  public static Schedule<? extends ISchedule> create(ISchedule newBackend) {
    return ApiWrapperUtil.getImplementingWrapper(Schedule.class, newBackend, "weka.dl4j.schedules");
  }

  @Override
  public T getBackend() {
    return backend;
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
