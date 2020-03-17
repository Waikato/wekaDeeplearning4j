package weka.dl4j.zoo;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.deeplearning4j.zoo.PretrainedType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.dl4j.Preferences;

import java.io.Serializable;
import java.util.*;

public abstract class AbstractZooModel implements OptionHandler, Serializable {

    protected PretrainedType m_pretrainedType = null;

    private org.deeplearning4j.zoo.ZooModel m_zooModelType;

    protected final Logger log = LoggerFactory.getLogger(AbstractZooModel.class);

    protected String m_layerToRemove, m_featureExtractionLayer, m_predictionLayerName = "predictions";

    protected int m_numFExtractOutputs;

    /**
     * Initialize the ZooModel as MLP
     *
     * @param numLabels Number of labels to adjust the output
     * @param seed Seed
     * @return MultiLayerNetwork of the specified ZooModel
     * @throws UnsupportedOperationException Init(...) was not supported (only CustomNet)
     */
    public abstract ComputationGraph init(int numLabels, long seed, int[] shape)
            throws UnsupportedOperationException;

    /**
     * Get the input shape of this zoomodel
     *
     * @return Input shape of this zoomodel
     */
    public abstract int[][] getShape();

    public ComputationGraph attemptToLoadWeights(org.deeplearning4j.zoo.ZooModel zooModel,
                                                 ComputationGraph defaultNet,
                                                 long seed,
                                                 int numLabels,
                                                 MultiLayerConfiguration conf,
                                                 int[] shape) {

        // If no pretrained weights specified, simply return the standard model
        if (m_pretrainedType == null)
            return defaultNet;

        // If the specified pretrained weights aren't available, return the standard model
        if (!checkPretrained(zooModel)) {
            m_pretrainedType = null;
            return defaultNet;
        }

        // If downloading the weights fails, return the standard model
        ComputationGraph pretrainedModel = downloadWeights(zooModel, conf, shape);
        if (pretrainedModel == null)
            return defaultNet;

        try {
            // Finally, create the transfer learning graph
            return new TransferLearning.GraphBuilder(pretrainedModel)
                    .fineTuneConfiguration(getFineTuneConfig(seed))
                    .removeVertexKeepConnections(m_layerToRemove)
                    .addLayer(m_predictionLayerName, createOutputLayer(numLabels), m_featureExtractionLayer)
                    .setOutputs(m_predictionLayerName).build();
        } catch (Exception ex) {
            ex.printStackTrace();
            log.error(pretrainedModel.summary());
            return defaultNet;
        }
    }


    public ComputationGraph attemptToLoadWeights(org.deeplearning4j.zoo.ZooModel zooModel,
                                                 ComputationGraph defaultNet,
                                                 long seed,
                                                 int numLabels) {
        return attemptToLoadWeights(zooModel, defaultNet, seed, numLabels, null, null);
    }

    public boolean isPretrained() {
        return m_pretrainedType != null;
    }

    protected FineTuneConfiguration getFineTuneConfig(long seed) {
        return new FineTuneConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(5e-5))
                .seed(seed)
                .build();
    }

    protected ComputationGraph downloadWeights(org.deeplearning4j.zoo.ZooModel net,
                                               MultiLayerConfiguration conf,
                                               int[] shape) {
        ComputationGraph cmpGraph = null;
        try {
            log.info(String.format("Downloading %s weights", m_pretrainedType));
//            Object pretrained = net.initPretrained(m_pretrainedType); TODO add support for MultiLayerNetwork models
//            if (pretrained instanceof MultiLayerNetwork) {
//                ComputationGraph thisGraph = mlpToCG(, shape);
//            }
            cmpGraph = (ComputationGraph) net.initPretrained(m_pretrainedType);
            return cmpGraph;
        } catch (Exception ex) {
            ex.printStackTrace();
            return cmpGraph;
        }
    }

    protected OutputLayer createOutputLayer(int numLabels) {
        return new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(m_numFExtractOutputs).nOut(numLabels)
                .weightInit(new NormalDistribution(0, 0.2 * (2.0 / (4096 + numLabels)))) //This weight init dist gave better results than Xavier
                .activation(Activation.SOFTMAX).build();
    }

    protected boolean checkPretrained(org.deeplearning4j.zoo.ZooModel dl4jModelType) {
        Set<PretrainedType> availableTypes = getAvailablePretrainedWeights(dl4jModelType);
        if (availableTypes.isEmpty()) {
            log.error("Sorry, no pretrained weights are available for this model");
            return false;
        }
        if (!availableTypes.contains(m_pretrainedType)){
            log.error(String.format("%s weights are not available for this model, " +
                    "please try one of: %s", m_pretrainedType, availableTypes.toString()));
            return false;
        }
        return true;
    }

    private Set<PretrainedType> getAvailablePretrainedWeights(org.deeplearning4j.zoo.ZooModel zooModel) {
        Set<PretrainedType> availableTypes = new HashSet<>();
        for (PretrainedType pretrainedType : PretrainedType.values()) {
            if (zooModel.pretrainedAvailable(pretrainedType)) {
                availableTypes.add(pretrainedType);
            }
        }
        return availableTypes;
    }

    public PretrainedType getPretrainedType() {
        return m_pretrainedType;
    }

    public AbstractZooModel setPretrainedType(PretrainedType pretrainedType) {
        throw new NotImplementedException();
    }

    protected AbstractZooModel setPretrainedType(PretrainedType pretrainedType,
                                                 int numFExtractOutputs,
                                                 String layerToRemove,
                                                 String featureExtractionLayer) {
        m_pretrainedType = pretrainedType;
        m_numFExtractOutputs = numFExtractOutputs;
        m_layerToRemove = layerToRemove;
        m_featureExtractionLayer = featureExtractionLayer;
        return this;
    }

    @OptionMetadata(
            description = "The name of the feature extraction layer in the model.",
            displayName = "Feature extraction layer",
            commandLineParamName = "extrac",
            commandLineParamSynopsis = "-extrac <String>",
            displayOrder = 0
    )
    public String getFeatureExtractionLayer() {
        return m_featureExtractionLayer;
    }

    public void setFeatureExtractionLayer(String featureExtractionLayer) {
        this.m_featureExtractionLayer = m_featureExtractionLayer;
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
