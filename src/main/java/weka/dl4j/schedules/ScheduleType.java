
package weka.dl4j.schedules;

import weka.dl4j.ApiWrapper;

/**
 * Proxy Enum for {@link org.nd4j.linalg.schedule.ScheduleType}. This is necessary as Weka's run
 * script cannot find the enum classes during the option parsing as they reside in the Dl4j backend
 * and are at that time not visible to the class loader.
 *
 * @author Steven Lang
 */
public enum ScheduleType implements ApiWrapper<org.nd4j.linalg.schedule.ScheduleType> {
  ITERATION,
  EPOCH;

  /**
   * Parse backend schedule type and return weka enum implementation.
   *
   * @param scheduleType Schedule type
   * @return Weka schedule type implementation
   */
  public static ScheduleType fromBackend(org.nd4j.linalg.schedule.ScheduleType scheduleType) {
    return valueOf(scheduleType.name());
  }

  public org.nd4j.linalg.schedule.ScheduleType getBackend() {
    switch (this) {
      case EPOCH:
        return org.nd4j.linalg.schedule.ScheduleType.EPOCH;
      case ITERATION:
        return org.nd4j.linalg.schedule.ScheduleType.ITERATION;
      default:
        throw new RuntimeException(
            "GradientNormalization method: " + this + " not found in Deeplearning4j backend.");
    }
  }

  @Override
  public void setBackend(org.nd4j.linalg.schedule.ScheduleType newBackend) {
    // Do nothing as this enum does not have a state
  }

  @Override
  public void initializeBackend() {
    // Do nothing as this enum does not have a state
  }
}
