package weka.dl4j.updater;

/**
 * A WEKA version of DeepLearning4j's ConstantScheduleImpl.
 *
 * @author Steven Lang
 */
public class NoOp extends Updater<org.nd4j.linalg.learning.config.NoOp> {
  private static final long serialVersionUID = 3503205924392465662L;

  @Override
  public void initializeBackend() {
    backend = new org.nd4j.linalg.learning.config.NoOp();
  }
}
