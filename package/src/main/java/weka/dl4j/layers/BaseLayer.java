// package weka.dl4j.layers;
//
// import org.deeplearning4j.nn.conf.GradientNormalization;
// import org.deeplearning4j.nn.conf.Updater;
// import org.deeplearning4j.nn.conf.distribution.Distribution;
// import org.deeplearning4j.nn.conf.inputs.InputType;
// import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
// import org.deeplearning4j.nn.weights.WeightInit;
// import org.nd4j.linalg.activations.IActivation;
// import weka.core.OptionMetadata;
// import weka.gui.ProgrammaticProperty;
//
// import java.util.Map;
//
/// **
// * An nd4j mini-batch iterator that iterates a given dataset.
// *
// * @author Steven Lang
// */
//
// public interface BaseLayer {
//
//    private static final long serialVersionUID = -3870462244324681550L;
//
//
//    @Override
//    @ProgrammaticProperty
//    default void setNIn(InputType inputType, boolean override) {
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public int getNIn() {
//        return super.getNIn();
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public WeightInit getWeightInit() {
//        return super.getWeightInit();
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public double getBiasInit() {
//        return super.getBiasInit();
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public Distribution getDist() {
//        return super.getDist();
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public double getLearningRate() {
//        return super.getLearningRate();
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public double getBiasLearningRate() {
//        return super.getBiasLearningRate();
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public Map<Integer, Double> getLearningRateSchedule() {
//        return super.getLearningRateSchedule();
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public double getMomentum() {
//        return super.getMomentum();
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public Map<Integer, Double> getMomentumSchedule() {
//        return super.getMomentumSchedule();
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public double getL1() {
//        return super.getL1();
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public double getL2() {
//        return super.getL2();
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public double getL1Bias() {
//        return super.getL1Bias();
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public double getL2Bias() {
//        return super.getL2Bias();
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public double getRho() {
//        return super.getRho();
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public double getEpsilon() {
//        return super.getEpsilon();
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public double getRmsDecay() {
//        return super.getRmsDecay();
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public double getAdamMeanDecay() {
//        return super.getAdamMeanDecay();
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public double getAdamVarDecay() {
//        return super.getAdamVarDecay();
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public GradientNormalization getGradientNormalization() {
//        return super.getGradientNormalization();
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public double getGradientNormalizationThreshold() {
//        return super.getGradientNormalizationThreshold();
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public void setWeightInit(WeightInit weightInit) {
//        super.setWeightInit(weightInit);
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public void setBiasInit(double biasInit) {
//        super.setBiasInit(biasInit);
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public void setDist(Distribution dist) {
//        super.setDist(dist);
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public void setLearningRate(double learningRate) {
//        super.setLearningRate(learningRate);
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public void setBiasLearningRate(double biasLearningRate) {
//        super.setBiasLearningRate(biasLearningRate);
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public void setLearningRateSchedule(Map<Integer, Double> learningRateSchedule) {
//        super.setLearningRateSchedule(learningRateSchedule);
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public void setMomentum(double momentum) {
//        super.setMomentum(momentum);
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public void setMomentumSchedule(Map<Integer, Double> momentumSchedule) {
//        super.setMomentumSchedule(momentumSchedule);
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public void setL1(double l1) {
//        super.setL1(l1);
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public void setL2(double l2) {
//        super.setL2(l2);
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public void setL1Bias(double l1Bias) {
//        super.setL1Bias(l1Bias);
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public void setL2Bias(double l2Bias) {
//        super.setL2Bias(l2Bias);
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public void setUpdater(Updater updater) {
//        super.setUpdater(updater);
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public void setRho(double rho) {
//        super.setRho(rho);
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public void setEpsilon(double epsilon) {
//        super.setEpsilon(epsilon);
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public void setRmsDecay(double rmsDecay) {
//        super.setRmsDecay(rmsDecay);
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public void setAdamMeanDecay(double adamMeanDecay) {
//        super.setAdamMeanDecay(adamMeanDecay);
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public void setAdamVarDecay(double adamVarDecay) {
//        super.setAdamVarDecay(adamVarDecay);
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public void setGradientNormalization(GradientNormalization gradientNormalization) {
//        super.setGradientNormalization(gradientNormalization);
//    }
//
//    @Override
//    @ProgrammaticProperty
//    public void setGradientNormalizationThreshold(double gradientNormalizationThreshold) {
//        super.setGradientNormalizationThreshold(gradientNormalizationThreshold);
//    }
// }
