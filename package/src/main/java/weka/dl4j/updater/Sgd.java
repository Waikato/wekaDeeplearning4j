package weka.dl4j.updater;

import weka.gui.ProgrammaticProperty;

/**
 * A WEKA version of DeepLearning4j's Sgd.
 *
 * @author Steven Lang
 */
public class Sgd extends org.nd4j.linalg.learning.config.Sgd implements Updater {

  private static final long serialVersionUID = 1852959048173443658L;

  @ProgrammaticProperty
  @Override
  public double getLearningRate() {
    return super.getLearningRate();
  }

  @ProgrammaticProperty
  @Override
  public void setLearningRate(double learningRate) {
    super.setLearningRate(learningRate);
  }
}
