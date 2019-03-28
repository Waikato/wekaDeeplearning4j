
package weka.dl4j.schedules;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Map;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import weka.core.Option;
import weka.core.OptionMetadata;
import weka.gui.ProgrammaticProperty;

/**
 * Map schedule for learning rates.
 *
 * @author Steven Lang
 */
@EqualsAndHashCode(callSuper = true)
@ToString
public class MapSchedule extends Schedule<org.nd4j.linalg.schedule.MapSchedule> {

  private static final long serialVersionUID = 3282040498224686346L;
  private Map<Integer, Double> values;

  @Override
  @Deprecated
  @ProgrammaticProperty
  public double getInitialValue() {
    return super.getInitialValue();
  }

  @Override
  @Deprecated
  @ProgrammaticProperty
  public void setInitialValue(double initialValue) {
    super.setInitialValue(initialValue);
  }


  @OptionMetadata(
      displayName = "mapValues",
      description = "The map values (default = {0: 0.1}).",
      commandLineParamName = "mapValues",
      commandLineParamSynopsis = "-mapValues <map<integer,double>>",
      displayOrder = 2
  )
  public Map<Integer, Double> getValues() {
    return values;
  }

  public void setValues(Map<Integer, Double> values) {
    this.values = values;
  }


  @Override
  public void initializeBackend() {
    values = Collections.singletonMap(0, 0.1);
    backend = new org.nd4j.linalg.schedule.MapSchedule(scheduleType.getBackend(), values);
  }

  @Override
  public void setBackend(org.nd4j.linalg.schedule.MapSchedule newBackend) {
    this.scheduleType = ScheduleType.fromBackend(newBackend.getScheduleType());
    this.values = newBackend.getValues();
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
