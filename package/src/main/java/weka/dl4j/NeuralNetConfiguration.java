package weka.dl4j;

import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.stepfunctions.StepFunction;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.gui.ProgrammaticProperty;

import java.io.Serializable;
import java.util.Enumeration;

/**
 * A version of DeepLearning4j's NeuralNetConfiguration that implements WEKA option handling.
 *
 * <p>The duplicate code of configuration parameters is necessary since the dl4j
 * NeuralNetConfiguration.Builder object is not serializable which is necessary for the weka GUI.
 *
 * @author Eibe Frank
 */
@EqualsAndHashCode
public class NeuralNetConfiguration implements Serializable, OptionHandler {

  private static final long serialVersionUID = -4384295102884151216L;

  protected WeightInit weightInit = WeightInit.XAVIER;
  protected double biasInit = 0.0;
  protected Distribution dist = null;
  protected double learningRate = 1e-1;
  protected double biasLearningRate = Double.NaN;
  protected double l1 = Double.NaN;
  protected double l2 = Double.NaN;
  protected double l1Bias = Double.NaN;
  protected double l2Bias = Double.NaN;
  protected double dropOut = 0;
  protected IUpdater iUpdater = new Sgd();
  protected double leakyreluAlpha = 0.01;
  protected boolean miniBatch = true;
  protected long seed = 0;
  protected boolean useRegularization = false;
  protected OptimizationAlgorithm optimizationAlgo =
      OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
  protected StepFunction stepFunction = null;
  protected boolean useDropConnect = false;
  protected boolean minimize = true;
  protected LearningRatePolicy learningRatePolicy = LearningRatePolicy.None;
  protected double lrPolicyDecayRate = Double.NaN;
  protected double lrPolicySteps = Double.NaN;
  protected double lrPolicyPower = Double.NaN;
  protected boolean pretrain = false;

  /** Constructor that provides default values for the settings. */
  public NeuralNetConfiguration() {
    setLearningRate(0.1);
    setUpdater(new weka.dl4j.updater.Adam());
    setWeightInit(WeightInit.XAVIER);
  }

  /**
   * Deliver access to the internal builder
   *
   * @return NeuralNetworkConfiguration
   */
  public org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder builder() {
    Builder builder = new Builder();
    builder
        .l1(l1)
        .l2(l2)
        .leakyreluAlpha(leakyreluAlpha)
        .learningRate(learningRate)
        .learningRateDecayPolicy(learningRatePolicy)
        .lrPolicyDecayRate(lrPolicyDecayRate)
        .lrPolicyPower(lrPolicyPower)
        .lrPolicySteps(lrPolicySteps)
        .optimizationAlgo(optimizationAlgo)
        .seed(seed)
        .stepFunction(stepFunction)
        .updater(iUpdater)
        .weightInit(weightInit)
        .dist(dist)
        .biasInit(biasInit)
        .biasLearningRate(biasLearningRate)
        .miniBatch(miniBatch)
        .minimize(minimize)
        .useDropConnect(useDropConnect);

    builder.setUseRegularization(useRegularization);
    builder.setPretrain(pretrain);
    return builder;
  }

  @OptionMetadata(
    displayName = "leaky relu alpha",
    description = "The parameter for the leaky relu (default = 0.1).",
    commandLineParamName = "leakyreluAlpha",
    commandLineParamSynopsis = "-leakyreluAlpha <double>",
    displayOrder = 0
  )
  public double getLeakyreluAlpha() {
    return leakyreluAlpha;
  }

  public void setLeakyreluAlpha(double a) {
    leakyreluAlpha = a;
  }

  @OptionMetadata(
    description =
        "Optimization algorithm (LINE_GRADIENT_DESCENT,"
            + " CONJUGATE_GRADIENT, HESSIAN_FREE, "
            + "LBFGS, STOCHASTIC_GRADIENT_DESCENT)",
    displayName = "optimization algorithm",
    commandLineParamName = "algorithm",
    commandLineParamSynopsis = "-algorithm <string>",
    displayOrder = 1
  )
  public OptimizationAlgorithm getOptimizationAlgo() {
    return optimizationAlgo;
  }

  public void setOptimizationAlgo(OptimizationAlgorithm optimAlgorithm) {
    optimizationAlgo = optimAlgorithm;
  }

  @OptionMetadata(
    displayName = "learning rate policy",
    description = "The learning rate policy (default = None).",
    commandLineParamName = "learningRatePolicy",
    commandLineParamSynopsis = "-learningRatePolicy <string>",
    displayOrder = 2
  )
  public LearningRatePolicy getLearningRatePolicy() {
    return learningRatePolicy;
  }

  public void setLearningRatePolicy(LearningRatePolicy p) {
    learningRatePolicy = p;
  }

  @OptionMetadata(
    displayName = "learning rate policy decay rate",
    description = "The learning rate policy decay rate (default = NaN).",
    commandLineParamName = "lrPolicyDecayRate",
    commandLineParamSynopsis = "-lrPolicyDecayRate <double>",
    displayOrder = 3
  )
  public double getLrPolicyDecayRate() {
    return lrPolicyDecayRate;
  }

  public void setLrPolicyDecayRate(double r) {
    lrPolicyDecayRate = r;
  }

  @OptionMetadata(
    displayName = "learning rate policy power",
    description = "The learning rate policy power (default = NaN).",
    commandLineParamName = "lrPolicyPower",
    commandLineParamSynopsis = "-lrPolicyPower <double>",
    displayOrder = 4
  )
  public double getLrPolicyPower() {
    return lrPolicyPower;
  }

  public void setLrPolicyPower(double r) {
    lrPolicyPower = r;
  }

  @OptionMetadata(
    displayName = "learning rate policy steps",
    description = "The learning rate policy steps (default = NaN).",
    commandLineParamName = "lrPolicySteps",
    commandLineParamSynopsis = "-lrPolicySteps <double>",
    displayOrder = 5
  )
  public double getLrPolicySteps() {
    return lrPolicySteps;
  }

  public void setLrPolicySteps(double r) {
    lrPolicySteps = r;
  }

  @OptionMetadata(
    displayName = "whether to minimize objective",
    description = "Whether to minimize objective.",
    commandLineParamIsFlag = true,
    commandLineParamName = "minimize",
    commandLineParamSynopsis = "-minimize",
    displayOrder = 7
  )
  public boolean isMinimize() {
    return minimize;
  }

  public void setMinimize(boolean b) {
    minimize = b;
  }

  @OptionMetadata(
    displayName = "whether to use drop connect",
    description = "Whether to use drop connect.",
    commandLineParamIsFlag = true,
    commandLineParamName = "useDropConnect",
    commandLineParamSynopsis = "-useDropConnect",
    displayOrder = 8
  )
  public boolean isUseDropConnect() {
    return useDropConnect;
  }

  public void setUseDropConnect(boolean b) {
    useDropConnect = b;
  }

  @OptionMetadata(
    displayName = "whether to use regularization",
    description = "Whether to use regularization.",
    commandLineParamIsFlag = true,
    commandLineParamName = "useRegularization",
    commandLineParamSynopsis = "-useRegularization",
    displayOrder = 9
  )
  public boolean isUseRegularization() {
    return useRegularization;
  }

  public void setUseRegularization(boolean b) {
    useRegularization = b;
  }

  @OptionMetadata(
    displayName = "step function",
    description = "The step function to use (default = default).",
    commandLineParamName = "stepFunction",
    commandLineParamSynopsis = "-stepFunction <string>",
    displayOrder = 11
  )
  public StepFunction getStepFunction() {
    return stepFunction;
  }

  public void setStepFunction(StepFunction f) {
    stepFunction = f;
  }

  @OptionMetadata(
    displayName = "updater",
    description = "The updater to use (default = SGD).",
    commandLineParamName = "updater",
    commandLineParamSynopsis = "-updater <string>",
    displayOrder = 12
  )
  public IUpdater getUpdater() {
    return iUpdater;
  }

  public void setUpdater(IUpdater updater) {
    iUpdater = updater;
  }

  @OptionMetadata(
    displayName = "learningrate",
    description = "The learningrate to use (default = 0.1).",
    commandLineParamName = "learningRate",
    commandLineParamSynopsis = "-learningRate <double>",
    displayOrder = 13
  )
  public double getLearningRate() {
    return learningRate;
  }

  public void setLearningRate(double lr) {
    learningRate = lr;
  }

  @OptionMetadata(
    displayName = "bias learning rate",
    description = "The bias learning rate (default = learningRate).",
    commandLineParamName = "biasLearningRate",
    commandLineParamSynopsis = "-biasLearningRate <double>",
    displayOrder = 14
  )
  public double getBiasLearningRate() {
    return this.biasLearningRate;
  }

  public void setBiasLearningRate(double biasLearningRate) {
    this.biasLearningRate = biasLearningRate;
  }

  @OptionMetadata(
    displayName = "l1 regularization factor",
    description = "L1 regularization factor (default = 0.00).",
    commandLineParamName = "l1",
    commandLineParamSynopsis = "-l1 <double>",
    displayOrder = 14
  )
  public double getL1() {
    return l1;
  }

  public void setL1(double l1) {
    this.l1 = l1;
  }

  @OptionMetadata(
    displayName = "l2 regularization factor",
    description = "L2 regularization factor (default = 0.00).",
    commandLineParamName = "l2",
    commandLineParamSynopsis = "-l2 <double>",
    displayOrder = 15
  )
  public double getL2() {
    return l2;
  }

  public void setL2(double l2) {
    this.l2 = l2;
  }

  @OptionMetadata(
    displayName = "weight initialization method",
    description = "The method for weight initialization (default = XAVIER).",
    commandLineParamName = "weightInit",
    commandLineParamSynopsis = "-weightInit <specification>",
    displayOrder = 18
  )
  public WeightInit getWeightInit() {
    return weightInit;
  }

  public void setWeightInit(WeightInit weightInit) {
    this.weightInit = weightInit;
  }

  @OptionMetadata(
    displayName = "distribution",
    description = "The distribution (default = NormalDistribution(1e-3, 1)).",
    commandLineParamName = "dist",
    commandLineParamSynopsis = "-dist <specification>",
    displayOrder = 19
  )
  public Distribution getDist() {
    return dist;
  }

  public void setDist(Distribution dist) {
    this.dist = dist;
  }

  @OptionMetadata(
    displayName = "bias initialization",
    description = "The bias initialization (default = 0.0).",
    commandLineParamName = "biasInit",
    commandLineParamSynopsis = "-biasInit <double>",
    displayOrder = 20
  )
  public double getBiasInit() {
    return this.biasInit;
  }

  public void setBiasInit(double biasInit) {
    this.biasInit = biasInit;
  }

  @ProgrammaticProperty
  public long getSeed() {
    return seed;
  }

  public void setSeed(long n) {
    seed = n;
  }

  @ProgrammaticProperty
  public boolean isMiniBatch() {
    return miniBatch;
  }

  public void setMiniBatch(boolean b) {
    miniBatch = b;
  }

  @ProgrammaticProperty
  public boolean isPretrain() {
    return pretrain;
  }

  public void setPretrain(boolean b) {
    pretrain = b;
  }

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
   * Returns a string describing this search method
   *
   * @return a description of the search method suitable for displaying in the explorer/experimenter
   *     gui
   */
  public String globalInfo() {
    return "Class for fine tuning configurations of the network.\n"
        + "Parameters set as NaN are unused.\n";
  }
}
