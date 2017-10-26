package weka.dl4j.layers;

import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.IActivation;
import weka.core.OptionMetadata;
import weka.gui.ProgrammaticProperty;

import java.util.Map;

/**
 * An nd4j mini-batch iterator that iterates a given dataset.
 *
 * @author Steven Lang
 */

public interface BaseLayer {

    /**
     * Set default parameters for this layer
     */
    default void setDefaults(){
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

    @OptionMetadata(
            displayName = "layer name",
            description = "The name of the layer (default = Hidden Layer).",
            commandLineParamName = "name", commandLineParamSynopsis = "-name <string>",
            displayOrder = 0)
    String getLayerName();

    void setLayerName(String layerName);

    @OptionMetadata(
            displayName = "number of units",
            description = "The number of units.",
            commandLineParamName = "nOut", commandLineParamSynopsis = "-nOut <int>",
            displayOrder = 1)
    int getNOut();

    void setNOut(int nOut);

    @OptionMetadata(
            displayName = "activation function",
            description = "The activation function to use (default = ReLU).",
            commandLineParamName = "activation", commandLineParamSynopsis = "-activation <specification>",
            displayOrder = 2)
    IActivation getActivationFunction();

    void setActivationFunction(IActivation activationFn);

    @ProgrammaticProperty
    IActivation getActivationFn();

    void setActivationFn(IActivation fn);

    @OptionMetadata(
            displayName = "weight initialization method",
            description = "The method for weight initialization (default = XAVIER).",
            commandLineParamName = "weightInit", commandLineParamSynopsis = "-weightInit <specification>",
            displayOrder = 3)
    WeightInit getWeightInit();

    void setWeightInit(WeightInit weightInit);

    @OptionMetadata(
            displayName = "bias initialization",
            description = "The bias initialization (default = 1.0).",
            commandLineParamName = "biasInit", commandLineParamSynopsis = "-biasInit <double>",
            displayOrder = 4)
    double getBiasInit();

    void setBiasInit(double biasInit);

    @OptionMetadata(
            displayName = "distribution",
            description = "The distribution (default = NormalDistribution(1e-3, 1)).",
            commandLineParamName = "dist", commandLineParamSynopsis = "-dist <specification>",
            displayOrder = 5)
    Distribution getDist();

    void setDist(Distribution dist);

    @OptionMetadata(
            displayName = "learning rate",
            description = "The learning rate (default = 0.01).",
            commandLineParamName = "lr", commandLineParamSynopsis = "-lr <double>",
            displayOrder = 6)
    double getLearningRate();

    void setLearningRate(double learningRate);

    @OptionMetadata(
            displayName = "bias learning rate",
            description = "The bias learning rate (default = 0.01).",
            commandLineParamName = "blr", commandLineParamSynopsis = "-blr <double>",
            displayOrder = 7)
    double getBiasLearningRate();

    void setBiasLearningRate(double biasLearningRate);

    @OptionMetadata(
            displayName = "learning rate schedule",
            description = "The learning rate schedule.",
            commandLineParamName = "lrSchedule", commandLineParamSynopsis = "-lrSchedule <specification>",
            displayOrder = 8)
    Map<Integer, Double> getLearningRateSchedule();

    void setLearningRateSchedule(Map<Integer, Double> learningRateSchedule);

    @OptionMetadata(
            displayName = "momentum",
            description = "The momentum (default = 0.9).",
            commandLineParamName = "momentum", commandLineParamSynopsis = "-momentum <double>",
            displayOrder = 9)
    double getMomentum();

    void setMomentum(double momentum);

    @OptionMetadata(
            displayName = "momentum schedule",
            description = "The momentum schedule.",
            commandLineParamName = "momentumSchedule", commandLineParamSynopsis = "-momentumSchedule <specification>",
            displayOrder = 10)
    Map<Integer, Double> getMomentumSchedule();

    void setMomentumSchedule(Map<Integer, Double> momentumSchedule);

    @OptionMetadata(
            displayName = "L1",
            description = "The L1 parameter (default = 0).",
            commandLineParamName = "L1", commandLineParamSynopsis = "-L1 <double>",
            displayOrder = 11)
    double getL1();

    void setL1(double l1);

    @OptionMetadata(
            displayName = "L2",
            description = "The L2 parameter (default = 0).",
            commandLineParamName = "L2", commandLineParamSynopsis = "-L2 <double>",
            displayOrder = 12)
    double getL2();

    void setL2(double l2);

    @OptionMetadata(
            displayName = "L1 bias",
            description = "The L1 bias parameter (default = 0).",
            commandLineParamName = "l1Bias", commandLineParamSynopsis = "-l1Bias <double>",
            displayOrder = 13)
    double getBiasL1();

    void setBiasL1(double biasL1);

    @OptionMetadata(
            displayName = "L2 bias",
            description = "The L2 bias parameter (default = 0).",
            commandLineParamName = "l2Bias", commandLineParamSynopsis = "-l2Bias <double>",
            displayOrder = 14)
    double getBiasL2();

    void setBiasL2(double biasL2);

    @OptionMetadata(
            displayName = "dropout parameter",
            description = "The dropout parameter (default = 0).",
            commandLineParamName = "dropout", commandLineParamSynopsis = "-dropout <double>",
            displayOrder = 15)
    double getDropOut();

    void setDropOut(double dropOut);

    @OptionMetadata(
            displayName = "updater for stochastic gradient descent",
            description = "The updater for stochastic gradient descent (default NESTEROVS).",
            commandLineParamName = "updater", commandLineParamSynopsis = "-updater <speficiation>",
            displayOrder = 16)
    Updater getUpdater();

    void setUpdater(Updater updater);

    @OptionMetadata(
            displayName = "ADADELTA's rho parameter",
            description = "ADADELTA's rho parameter (default = 0).",
            commandLineParamName = "rho", commandLineParamSynopsis = "-rho <double>",
            displayOrder = 17)
    double getRho();

    void setRho(double rho);

    @OptionMetadata(
            displayName = "ADADELTA's epsilon parameter",
            description = "ADADELTA's epsilon parameter (default = 1e-6).",
            commandLineParamName = "epsilon", commandLineParamSynopsis = "-epsilon <double>",
            displayOrder = 18)
    double getEpsilon();

    void setEpsilon(double epsilon);

    @OptionMetadata(
            displayName = "RMSPROP's RMS decay parameter",
            description = "RMSPROP's RMS decay parameter (default = 0.95).",
            commandLineParamName = "rmsDecay", commandLineParamSynopsis = "-rmsDecay <double>",
            displayOrder = 19)
    double getRmsDecay();

    void setRmsDecay(double rmsDecay);

    @OptionMetadata(
            displayName = "ADAM's mean decay parameter",
            description = "ADAM's mean decay parameter (default 0.9).",
            commandLineParamName = "adamMeanDecay", commandLineParamSynopsis = "-adamMeanDecay <double>",
            displayOrder = 20)
    double getAdamMeanDecay();

    void setAdamMeanDecay(double adamMeanDecay);

    @OptionMetadata(
            displayName = "ADAMS's var decay parameter",
            description = "ADAM's var decay parameter (default 0.999).",
            commandLineParamName = "adamVarDecay", commandLineParamSynopsis = "-adamVarDecay <double>",
            displayOrder = 21)
    double getAdamVarDecay();

    void setAdamVarDecay(double adamVarDecay);

    @OptionMetadata(
            displayName = "gradient normalization method",
            description = "The gradient normalization method (default = None).",
            commandLineParamName = "gradientNormalization", commandLineParamSynopsis = "-gradientNormalization <specification>",
            displayOrder = 22)
    GradientNormalization getGradientNormalization();

    void setGradientNormalization(GradientNormalization gradientNormalization);

    @OptionMetadata(
            displayName = "gradient normalization threshold",
            description = "The gradient normalization threshold (default = 1).",
            commandLineParamName = "gradNormThreshold", commandLineParamSynopsis = "-gradNormThreshold <double>",
            displayOrder = 23)
    double getGradientNormalizationThreshold();

    void setGradientNormalizationThreshold(double gradientNormalizationThreshold);

    @ProgrammaticProperty
    double getL1Bias();

    void setL1Bias(int l1bias);

    @ProgrammaticProperty
    double getL2Bias();

    void setL2Bias(int l2bias);
}
