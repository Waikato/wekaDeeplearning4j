package weka.dl4j.weightnoise;

import java.io.Serializable;
import java.util.Enumeration;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import org.nd4j.linalg.schedule.ISchedule;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.dl4j.schedules.Schedule;

/**
 * DropConnect wrapper.
 *
 * @author Steven Lang
 */
@EqualsAndHashCode(callSuper = true)
@ToString
public class DropConnect extends AbstractWeightNoise<org.deeplearning4j.nn.conf.weightnoise.DropConnect> implements OptionHandler, Serializable {

  private static final long serialVersionUID = -8153032592556766929L;


  /**
   * Get the learning rate schedule
   *
   * @return Learning rate schedule
   */
  @OptionMetadata(
      displayName = "weightRetainProb",
      description = "The weightRetainProb (default = 0.0).",
      commandLineParamName = "weightRetainProb",
      commandLineParamSynopsis = "-weightRetainProb <double>",
      displayOrder = 1
  )
  public double getWeightRetainProbability(){
    return backend.getWeightRetainProb();
  }

  public void setWeightRetainProbability(double prob){
    backend.setWeightRetainProb(prob);
  }

  /**
   * Get the learning rate schedule
   *
   * @return Learning rate schedule
   */
  @OptionMetadata(
      displayName = "weightRetainProbSchedule",
      description = "The weight retain probability schedule (default = ConstantScheduleImpl).",
      commandLineParamName = "weightRetainProbSchedule",
      commandLineParamSynopsis = "-weightRetainProbSchedule <Schedule>",
      displayOrder = 2
  )
  public Schedule<? extends ISchedule> getWeightRetainProbabilitySchedule(){
    return Schedule.create(backend.getWeightRetainProbSchedule());
  }

  public void setWeightRetainProbabilitySchedule(Schedule<? extends ISchedule> sched) {
    backend.setWeightRetainProbSchedule(sched.getBackend());
  }


  @Override
  public void initializeBackend() {
    backend=new org.deeplearning4j.nn.conf.weightnoise.DropConnect(0.0);
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
