package weka.dl4j;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.stepfunctions.StepFunction;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.dl4j.stepfunctions.NegativeGradientStepFunction;
import weka.gui.ProgrammaticProperty;
import java.io.Serializable;

import java.util.Enumeration;
import java.util.Map;

/**
 * A version of DeepLearning4j's NeuralNetConfiguration that implements WEKA option handling.
 *
 * @author Eibe Frank
 *
 * @version $Revision: 11711 $
 */
public class NeuralNetConfiguration extends org.deeplearning4j.nn.conf.NeuralNetConfiguration implements Serializable, OptionHandler {

  private static final long serialVersionUID = -4384295102884151216L;
  /**
   * Internal configuration builder
   */
  private Builder builder = new Builder();

  /**
   * Constructor that provides default values for the settings.
   */
  public NeuralNetConfiguration() {
    builder = new Builder();

    setUpdater(Updater.ADAM);

    this.leakyreluAlpha = 0.01D;
    this.miniBatch = true;
    this.numIterations = 1;
    this.maxNumLineSearchIterations = 5;
    this.useRegularization = true; // Changed this from the default in deepLearning4j.
    this.optimizationAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
    this.stepFunction = new NegativeGradientStepFunction();
    this.useDropConnect = false;
    this.minimize = true;
    this.learningRatePolicy = LearningRatePolicy.None;
    this.lrPolicyDecayRate = 0.0D / 0.0;
    this.lrPolicySteps = 0.0D / 0.0;
    this.lrPolicyPower = 0.0D / 0.0;
    this.pretrain = false;
  }

  /**
   * Deliver access to the internal builder
   * @return NeuralNetworkConfiguration
   */
  public org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder builder(){
    return builder;
  }

  @OptionMetadata(
          displayName = "leaky relu alpha",
          description = "The parameter for the leaky relu (default = 0.1).",
          commandLineParamName = "leakyreluAlpha", commandLineParamSynopsis = "-leakyreluAlpha <double>",
          displayOrder = 1)
  public double getLeakyreluAlpha() { return super.getLeakyreluAlpha(); }

  public void setLeakyreluAlpha(double a) {
    builder.leakyreluAlpha(a);
  }
  @OptionMetadata(
          description = "Optimization algorithm (LINE_GRADIENT_DESCENT,"
                  + " CONJUGATE_GRADIENT, HESSIAN_FREE, "
                  + "LBFGS, STOCHASTIC_GRADIENT_DESCENT)",
          displayName = "optimization algorithm", commandLineParamName = "algorithm",
          commandLineParamSynopsis = "-algorithm <string>", displayOrder = 0)
  public OptimizationAlgorithm getOptimizationAlgo() {
    return builder.getOptimizationAlgo();
  }

  public void setOptimizationAlgo(OptimizationAlgorithm optimAlgorithm) {
    builder.optimizationAlgo(optimAlgorithm);
  }

  @Override
  public void setL1ByParam(Map<String, Double> l1ByParam) {
    super.setL1ByParam(l1ByParam);
  }
  
  @OptionMetadata(
          displayName = "learning rate policy",
          description = "The learning rate policy (default = None).",
          commandLineParamName = "learningRatePolicy", commandLineParamSynopsis = "-learningRatePolicy <string>",
          displayOrder = 2)
  public LearningRatePolicy getLearningRatePolicy() { return super.getLearningRatePolicy(); }
  public void setLearningRatePolicy(LearningRatePolicy p) {
    builder.learningRateDecayPolicy(p);
  }

  @OptionMetadata(
          displayName = "learning rate policy decay rate",
          description = "The learning rate policy decay rate (default = NaN).",
          commandLineParamName = "lrPolicyDecayRate", commandLineParamSynopsis = "-lrPolicyDecayRate <double>",
          displayOrder = 3)
  public double getLrPolicyDecayRate() { return super.getLrPolicyDecayRate(); }
  public void setLrPolicyDecayRate(double r) { builder.lrPolicyDecayRate(r); }

  @OptionMetadata(
          displayName = "learning rate policy power",
          description = "The learning rate policy power (default = NaN).",
          commandLineParamName = "lrPolicyPower", commandLineParamSynopsis = "-lrPolicyPower <double>",
          displayOrder = 4)
  public double getLrPolicyPower() { return super.getLrPolicyPower(); }
  public void setLrPolicyPower(double r) { builder.lrPolicyPower(r); }

  @OptionMetadata(
          displayName = "learning rate policy steps",
          description = "The learning rate policy steps (default = NaN).",
          commandLineParamName = "lrPolicySteps", commandLineParamSynopsis = "-lrPolicySteps <double>",
          displayOrder = 5)
  public double getLrPolicySteps() { return super.getLrPolicySteps(); }
  public void setLrPolicySteps(double r) { builder.lrPolicySteps(r); }

  @OptionMetadata(
          displayName = "maximum number of line search iterations",
          description = "The maximum number of line search iterations (default = 5).",
          commandLineParamName = "maxNumLineSearchIterations", commandLineParamSynopsis = "-maxNumLineSearchIterations <int>",
          displayOrder = 6)
  public int getMaxNumLineSearchIterations() { return super.getMaxNumLineSearchIterations(); }
  public void setMaxNumLineSearchIterations(int n) { builder.maxNumLineSearchIterations(n); }

  @OptionMetadata(
          displayName = "whether to minimize objective",
          description = "Whether to minimize objective.", commandLineParamIsFlag = true,
          commandLineParamName = "minimize", commandLineParamSynopsis = "-minimize",
          displayOrder = 7)
  public boolean isMinimize() { return super.isMinimize(); }
  public void setMinimize(boolean b) { builder.minimize(b); }

  @OptionMetadata(
          displayName = "whether to use drop connect",
          description = "Whether to use drop connect.", commandLineParamIsFlag = true,
          commandLineParamName = "useDropConnect", commandLineParamSynopsis = "-useDropConnect",
          displayOrder = 8)
  public boolean isUseDropConnect() { return super.isUseDropConnect(); }
  public void setUseDropConnect(boolean b) { builder.useDropConnect(b); }

  @OptionMetadata(
          displayName = "whether to use regularization",
          description = "Whether to use regularization.", commandLineParamIsFlag = true,
          commandLineParamName = "useRegularization", commandLineParamSynopsis = "-useRegularization",
          displayOrder = 9)
  public boolean isUseRegularization() { return super.isUseRegularization(); }
  public void setUseRegularization(boolean b) { builder.regularization(b); }

  @OptionMetadata(
          displayName = "number of iterations for optimization",
          description = "The number of iterations for optimization (default = 1).",
          commandLineParamName = "numIterations", commandLineParamSynopsis = "-numIterations <int>",
          displayOrder = 10)
  public int getNumIterations() { return super.getNumIterations(); }
  public void setNumIterations(int n) { builder.iterations(n); }

  @OptionMetadata(
          displayName = "step function",
          description = "The step function to use (default = default).",
          commandLineParamName = "stepFunction", commandLineParamSynopsis = "-stepFunction <string>",
          displayOrder = 11)
  public StepFunction getStepFunction() { return super.getStepFunction(); }
  public void setStepFunction(StepFunction f) { builder.stepFunction(f); }

  @OptionMetadata(
          displayName = "updater",
          description = "The updater to use (default = SGD).",
          commandLineParamName = "updater", commandLineParamSynopsis = "-updater <string>",
          displayOrder = 12)
  public Updater getUpdater() { return builder.getUpdater(); }
  public void setUpdater(Updater updater) {
    builder.updater(updater);
  }
  @OptionMetadata(
          displayName = "learning rate",
          description = "The global learning rate (default = 0.01).",
          commandLineParamName = "lr", commandLineParamSynopsis = "-lr <double>",
          displayOrder = 13)
  public double getLearningRate() { return builder.getLearningRate(); }
  public void setLearningRate(double lr) {
    builder.learningRate(lr);
  }

  @OptionMetadata(
          displayName = "l1 regularization factor",
          description = "L1 regularization factor (default = 0.00).",
          commandLineParamName = "l1", commandLineParamSynopsis = "-l1 <double>",
          displayOrder = 14)
  public double getL1() { return builder.getL1(); }
  public void setL1(double l1) {
    builder.l1(l1);
  }

  @OptionMetadata(
          displayName = "l2 regularization factor",
          description = "L2 regularization factor (default = 0.00).",
          commandLineParamName = "l2", commandLineParamSynopsis = "-l2 <double>",
          displayOrder = 15)
  public double getL2() { return builder.getL2(); }
  public void setL2(double l2) {
    builder.l2(l2);
  }

  // Not working as of dl4j 0.9.1
//
//  @OptionMetadata(
//          displayName = "L1 bias",
//          description = "The L1 bias parameter (default = 0).",
//          commandLineParamName = "l1Bias", commandLineParamSynopsis = "-l1Bias <double>",
//          displayOrder = 16)
//  public double getBiasL1() {
//    return builder.getL1Bias();
//  }
//  public void setBiasL1(double biasL1) {
//    builder.l1Bias(biasL1);
//  }
//
//  @OptionMetadata(
//          displayName = "L2 bias",
//          description = "The L2 bias parameter (default = 0).",
//          commandLineParamName = "l2Bias", commandLineParamSynopsis = "-l2Bias <double>",
//          displayOrder = 17)
//  public double getBiasL2() {
//    return builder.getL2Bias();
//  }
//  public void setBiasL2(double biasL2) {
//    builder.l2Bias(biasL2);
//  }

  @OptionMetadata(
          displayName = "weight initialization method",
          description = "The method for weight initialization (default = XAVIER).",
          commandLineParamName = "weightInit", commandLineParamSynopsis = "-weightInit <specification>",
          displayOrder = 18)
  public WeightInit getWeightInit() {
    return builder.getWeightInit();
  }
  public void setWeightInit(WeightInit weightInit) {
    builder.weightInit(weightInit);
  }


  
  @ProgrammaticProperty
  public int getIterationCount() { return super.getIterationCount(); }
  public void setIterationCount(int n) { super.setIterationCount(n); }

  @ProgrammaticProperty
  public long getSeed() { return super.getSeed(); }
  public void setSeed(long n) { builder.seed(n); }

  @ProgrammaticProperty
  public boolean isMiniBatch() { return super.isMiniBatch(); }
  public void setMiniBatch(boolean b) { builder.miniBatch(b); }

  @ProgrammaticProperty
  public boolean isPretrain() { return super.isPretrain(); }
  public void setPretrain(boolean b) { builder.setPretrain(b); }
  

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration<Option> listOptions() {

    return Option.listOptionsForClass(this.getClass()).elements();
  }

  /**
   * Gets the current settings of the Classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String[] getOptions() {

    return Option.getOptions(this, this.getClass());
  }

  /**
   * Parses a given list of options.
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {

    Option.setOptions(options, this, this.getClass());
  }

  /**
   * Dummy builder class
   */
  public static class Builder extends org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder implements Serializable{

  }
}
