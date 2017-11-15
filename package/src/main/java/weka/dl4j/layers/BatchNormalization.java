/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    ConvolutionLayer.java
 *    Copyright (C) 2016 University of Waikato, Hamilton, New Zealand
 *
 */
package weka.dl4j.layers;

import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.IActivation;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.dl4j.activations.ActivationIdentity;
import weka.gui.ProgrammaticProperty;

import java.io.Serializable;
import java.util.Enumeration;
import java.util.Map;

/**
 * A version of DeepLearning4j's BatchNormalization layer that implements WEKA option handling.
 *
 * @author Christopher Beckham
 * @author Eibe Frank
 */
public class BatchNormalization extends org.deeplearning4j.nn.conf.layers.BatchNormalization
    implements OptionHandler, Serializable {

  /** The ID used to serialize this class. */
  private static final long serialVersionUID = 6804344091980568487L;

  /** Constructor for setting some defaults. */
  public BatchNormalization() {
    setLayerName("Batch normalization layer");
    setActivationFunction(new ActivationIdentity());
    setLearningRate(Double.NaN);
    setBiasLearningRate(Double.NaN);
    setMomentum(Double.NaN);
    setBiasInit(Double.NaN);
    setAdamMeanDecay(Double.NaN);
    setAdamVarDecay(Double.NaN);
    setEpsilon(Double.NaN);
    setRmsDecay(Double.NaN);
    setL1(Double.NaN);
    setL2(Double.NaN);
    setRho(Double.NaN);
  }

  /**
   * Global info.
   *
   * @return string describing this class.
   */
  public String globalInfo() {
    return "A convolution layer from DeepLearning4J.";
  }

  @OptionMetadata(
    displayName = "layer name",
    description = "The name of the layer (default = Batch normalization Layer).",
    commandLineParamName = "name",
    commandLineParamSynopsis = "-name <string>",
    displayOrder = 0
  )
  public String getLayerName() {
    return this.layerName;
  }

  public void setLayerName(String layerName) {
    this.layerName = layerName;
  }

  @OptionMetadata(
    displayName = "decay parameter",
    description = "The decay parameter (default = 0.9).",
    commandLineParamName = "decay",
    commandLineParamSynopsis = "-decay <double>",
    displayOrder = 1
  )
  public double getDecay() {
    return this.decay;
  }

  public void setDecay(double decay) {
    this.decay = decay;
  }

  @OptionMetadata(
    displayName = "eps parameter",
    description = "The eps parameter (default = 1e-5).",
    commandLineParamName = "eps",
    commandLineParamSynopsis = "-eps <double>",
    displayOrder = 2
  )
  public double getEps() {
    return this.eps;
  }

  public void setEps(double eps) {
    this.eps = eps;
  }

  @OptionMetadata(
    displayName = "gamma parameter",
    description = "The gamma parameter (default = 1).",
    commandLineParamName = "gamma",
    commandLineParamSynopsis = "-gamma <double>",
    displayOrder = 3
  )
  public double getGamma() {
    return this.gamma;
  }

  public void setGamma(double gamma) {
    this.gamma = gamma;
  }

  @OptionMetadata(
    displayName = "beta parameter",
    description = "The beta parameter (default = 0).",
    commandLineParamName = "beta",
    commandLineParamSynopsis = "-beta <double>",
    displayOrder = 4
  )
  public double getBeta() {
    return this.beta;
  }

  public void setBeta(double beta) {
    this.beta = beta;
  }

  @OptionMetadata(
    displayName = "lock gamma and beta",
    description = "Whether to lock gamma and beta.",
    commandLineParamName = "beta",
    commandLineParamSynopsis = "-lockGammaBeta",
    displayOrder = 5
  )
  public boolean getLockGammaAndBeta() {
    return super.isLockGammaBeta();
  }

  public void setLockGammaAndBeta(boolean lgb) {
    super.setLockGammaBeta(lgb);
  }

  @ProgrammaticProperty
  public boolean isLockGammaBeta() {
    return super.isLockGammaBeta();
  }

  public void setLockGammaBeta(boolean lgb) {
    super.setLockGammaBeta(lgb);
  }

  @OptionMetadata(
    displayName = "noMinibatch",
    description = "Whether minibatches are not not used.",
    commandLineParamName = "noMinibatch",
    commandLineParamSynopsis = "-noMinibatch",
    displayOrder = 6
  )
  public boolean getNoMinibatch() {
    return !super.isMinibatch();
  }

  public void setNoMinibatch(boolean b) {
    super.setMinibatch(!b);
  }

  @ProgrammaticProperty
  public boolean isMinibatch() {
    return super.isMinibatch();
  }

  public void setMinibatch(boolean b) {
    super.setMinibatch(b);
  }

  @OptionMetadata(
    displayName = "activation function",
    description = "The activation function to use (default = Identity).",
    commandLineParamName = "activation",
    commandLineParamSynopsis = "-activation <specification>",
    displayOrder = 7
  )
  public IActivation getActivationFunction() {
    return this.activationFn;
  }

  public void setActivationFunction(IActivation activationFn) {
    this.activationFn = activationFn;
  }

  @ProgrammaticProperty
  public IActivation getActivationFn() {
    return super.getActivationFn();
  }

  public void setActivationFn(IActivation fn) {
    super.setActivationFn(fn);
  }

  @OptionMetadata(
    displayName = "dropout parameter",
    description = "The dropout parameter (default = 0).",
    commandLineParamName = "dropout",
    commandLineParamSynopsis = "-dropout <double>",
    displayOrder = 15
  )
  public double getDropOut() {
    return this.dropOut;
  }

  public void setDropOut(double dropOut) {
    this.dropOut = dropOut;
  }

  @ProgrammaticProperty
  @Deprecated
  public WeightInit getWeightInit() {
    return this.weightInit;
  }

  @ProgrammaticProperty
  @Deprecated
  public void setWeightInit(WeightInit weightInit) {
    this.weightInit = weightInit;
  }

  @ProgrammaticProperty
  @Deprecated
  public double getBiasInit() {
    return this.biasInit;
  }

  @ProgrammaticProperty
  @Deprecated
  public void setBiasInit(double biasInit) {
    this.biasInit = biasInit;
  }

  @ProgrammaticProperty
  @Deprecated
  public Distribution getDist() {
    return this.dist;
  }

  @ProgrammaticProperty
  @Deprecated
  public void setDist(Distribution dist) {
    this.dist = dist;
  }

  @ProgrammaticProperty
  @Deprecated
  public double getLearningRate() {
    return this.learningRate;
  }

  @ProgrammaticProperty
  @Deprecated
  public void setLearningRate(double learningRate) {
    this.learningRate = learningRate;
  }

  @ProgrammaticProperty
  @Deprecated
  public double getBiasLearningRate() {
    return this.biasLearningRate;
  }

  @ProgrammaticProperty
  @Deprecated
  public void setBiasLearningRate(double biasLearningRate) {
    this.biasLearningRate = biasLearningRate;
  }

  @ProgrammaticProperty
  @Deprecated
  public Map<Integer, Double> getLearningRateSchedule() {
    return this.learningRateSchedule;
  }

  @ProgrammaticProperty
  @Deprecated
  public void setLearningRateSchedule(Map<Integer, Double> learningRateSchedule) {
    this.learningRateSchedule = learningRateSchedule;
  }

  @ProgrammaticProperty
  @Deprecated
  public double getMomentum() {
    return this.momentum;
  }

  @ProgrammaticProperty
  @Deprecated
  public void setMomentum(double momentum) {
    this.momentum = momentum;
  }

  @ProgrammaticProperty
  @Deprecated
  public Map<Integer, Double> getMomentumSchedule() {
    return this.momentumSchedule;
  }

  @ProgrammaticProperty
  @Deprecated
  public void setMomentumSchedule(Map<Integer, Double> momentumSchedule) {
    this.momentumSchedule = momentumSchedule;
  }

  @ProgrammaticProperty
  @Deprecated
  public double getL1() {
    return this.l1;
  }

  @ProgrammaticProperty
  @Deprecated
  public void setL1(double l1) {
    this.l1 = l1;
  }

  @ProgrammaticProperty
  @Deprecated
  public double getL2() {
    return this.l2;
  }

  @ProgrammaticProperty
  @Deprecated
  public void setL2(double l2) {
    this.l2 = l2;
  }

  @ProgrammaticProperty
  @Deprecated
  public double getBiasL1() {
    return this.l1Bias;
  }

  @ProgrammaticProperty
  @Deprecated
  public void setBiasL1(double biasL1) {
    this.l1Bias = biasL1;
  }

  @ProgrammaticProperty
  @Deprecated
  public double getBiasL2() {
    return this.l2Bias;
  }

  @ProgrammaticProperty
  @Deprecated
  public void setBiasL2(double biasL2) {
    this.l2Bias = biasL2;
  }

  @ProgrammaticProperty
  @Deprecated
  public Updater getUpdater() {
    return this.updater;
  }

  @ProgrammaticProperty
  @Deprecated
  public void setUpdater(Updater updater) {
    this.updater = updater;
  }

  @ProgrammaticProperty
  @Deprecated
  public double getRho() {
    return this.rho;
  }

  @ProgrammaticProperty
  @Deprecated
  public void setRho(double rho) {
    this.rho = rho;
  }

  @ProgrammaticProperty
  @Deprecated
  public double getEpsilon() {
    return this.epsilon;
  }

  @ProgrammaticProperty
  @Deprecated
  public void setEpsilon(double epsilon) {
    this.epsilon = epsilon;
  }

  @ProgrammaticProperty
  @Deprecated
  public double getRmsDecay() {
    return this.rmsDecay;
  }

  @ProgrammaticProperty
  @Deprecated
  public void setRmsDecay(double rmsDecay) {
    this.rmsDecay = rmsDecay;
  }

  @ProgrammaticProperty
  @Deprecated
  public double getAdamMeanDecay() {
    return this.adamMeanDecay;
  }

  public void setAdamMeanDecay(double adamMeanDecay) {
    this.adamMeanDecay = adamMeanDecay;
  }

  @ProgrammaticProperty
  @Deprecated
  public double getAdamVarDecay() {
    return this.adamVarDecay;
  }

  @ProgrammaticProperty
  @Deprecated
  public void setAdamVarDecay(double adamVarDecay) {
    this.adamVarDecay = adamVarDecay;
  }

  @ProgrammaticProperty
  @Deprecated
  public GradientNormalization getGradientNormalization() {
    return this.gradientNormalization;
  }

  @ProgrammaticProperty
  @Deprecated
  public void setGradientNormalization(GradientNormalization gradientNormalization) {
    this.gradientNormalization = gradientNormalization;
  }

  @ProgrammaticProperty
  @Deprecated
  public double getGradientNormalizationThreshold() {
    return this.gradientNormalizationThreshold;
  }

  @ProgrammaticProperty
  @Deprecated
  public void setGradientNormalizationThreshold(double gradientNormalizationThreshold) {
    this.gradientNormalizationThreshold = gradientNormalizationThreshold;
  }

  @ProgrammaticProperty
  @Deprecated
  public int getNIn() {
    return super.getNIn();
  }

  public void setNIn(int nIn) {
    this.nIn = nIn;
  }

  @ProgrammaticProperty
  @Deprecated
  public int getNOut() {
    return super.getNOut();
  }

  public void setNOut(int nOut) {
    this.nOut = nOut;
  }

  @ProgrammaticProperty
  @Deprecated
  public double getL1Bias() {
    return super.getL1Bias();
  }

  public void setL1Bias(int l1bias) {
    super.setL1Bias(l1bias);
  }

  @ProgrammaticProperty
  @Deprecated
  public double getL2Bias() {
    return super.getL2Bias();
  }

  public void setL2Bias(int l2bias) {
    super.setL2Bias(l2bias);
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
