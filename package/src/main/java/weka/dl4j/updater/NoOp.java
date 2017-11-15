package weka.dl4j.updater;

import weka.gui.ProgrammaticProperty;

/**
 * A WEKA version of DeepLearning4j's NoOp.
 *
 * @author Steven Lang
 */
public class NoOp extends org.nd4j.linalg.learning.config.NoOp implements Updater {
  private static final long serialVersionUID = 3503205924392465662L;

  @Override
  @ProgrammaticProperty
  public double getLearningRate() {
    return 0;
  }

  @Override
  @ProgrammaticProperty
  public void setLearningRate(double learningRate) {}
}
