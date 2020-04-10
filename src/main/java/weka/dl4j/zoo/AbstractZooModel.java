package weka.dl4j.zoo;

import lombok.extern.log4j.Log4j2;
import org.apache.commons.lang.NotImplementedException;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.dl4j.PretrainedType;

import java.io.Serializable;
import java.util.*;

/**
 *
 * @author Rhys Compton
 */
@Log4j2
public abstract class AbstractZooModel implements OptionHandler, Serializable {

    protected weka.dl4j.PretrainedType m_pretrainedType = PretrainedType.NONE;

    private org.deeplearning4j.zoo.ZooModel m_zooModelType;

    protected String m_outputLayer, m_featureExtractionLayer, m_predictionLayerName = "weka_predictions";

    protected String[] m_extraLayersToRemove = new String[0];

    protected int m_numFExtractOutputs;

    private long seed, numLabels;

    /**
     * Initialize the ZooModel as MLP
     *
     * @param numLabels Number of labels to adjust the output
     * @param seed Seed
     * @param shape shape
     * @param filterMode True if creating for feature extraction
     * @return MultiLayerNetwork of the specified ZooModel
     * @throws UnsupportedOperationException Init(...) was not supported (only CustomNet)
     */
    public abstract ComputationGraph init(int numLabels, long seed, int[] shape, boolean filterMode)
            throws UnsupportedOperationException;

    /**
     * Get the input shape of this zoomodel
     *
     * @return Input shape of this zoomodel
     */
    public abstract int[][] getShape();

    public void setVariation(Enum var) {
        throw new NotImplementedException();
    }

    public ComputationGraph attemptToLoadWeights(org.deeplearning4j.zoo.ZooModel zooModel,
                                                 ComputationGraph defaultNet,
                                                 long seed,
                                                 int numLabels,
                                                 boolean filterMode) {

        this.seed = seed;
        this.numLabels = numLabels;

        // If no pretrained weights specified, simply return the standard model
        if (m_pretrainedType == PretrainedType.NONE)
            return defaultNet;

        // If the specified pretrained weights aren't available, return the standard model
        if (!checkPretrained(zooModel)) {
            m_pretrainedType = PretrainedType.NONE;
            return defaultNet;
        }

        // If downloading the weights fails, return the standard model
        ComputationGraph pretrainedModel = downloadWeights(zooModel);
        if (pretrainedModel == null)
            return defaultNet;

        System.out.println(pretrainedModel.summary());

        if (!filterMode)
            return addFinalOutputLayer(pretrainedModel);
        else
            return pretrainedModel;
    }

    protected ComputationGraph addFinalOutputLayer(ComputationGraph computationGraph, long seed, int numLabels) {
        this.seed = seed;
        this.numLabels = numLabels;
        return addFinalOutputLayer(computationGraph);
    }

    protected ComputationGraph addFinalOutputLayer(ComputationGraph computationGraph) {
        try {
            TransferLearning.GraphBuilder graphBuilder = new TransferLearning.GraphBuilder(computationGraph)
                    .fineTuneConfiguration(getFineTuneConfig())
                    .addLayer(m_predictionLayerName, createOutputLayer(), m_featureExtractionLayer)
                    .setOutputs(m_predictionLayerName);

            // Remove the old output layer, but keep the connections
            graphBuilder.removeVertexKeepConnections(m_outputLayer);
            // Remove any layers we don't want
            for (String layer : m_extraLayersToRemove) {
                graphBuilder.removeVertexAndConnections(layer);
            }

            System.out.println(graphBuilder.build().summary());
            return graphBuilder.build();
        } catch (Exception ex) {
            ex.printStackTrace();
            log.error(computationGraph.summary());
            return computationGraph;
        }

    }

    public boolean isPretrained() {
        return m_pretrainedType != PretrainedType.NONE;
    }

    protected FineTuneConfiguration getFineTuneConfig() {
        return new FineTuneConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(5e-5))
                .seed(seed)
                .build();
    }

    protected ComputationGraph downloadWeights(org.deeplearning4j.zoo.ZooModel zooModel) {
        try {
            log.info(String.format("Downloading %s weights", m_pretrainedType));
            Object pretrained = zooModel.initPretrained(m_pretrainedType.getBackend());
            if (pretrained instanceof MultiLayerNetwork) {
                return ((MultiLayerNetwork) pretrained).toComputationGraph();
            } else {
                return (ComputationGraph) pretrained;
            }
        } catch (Exception ex) {
            ex.printStackTrace();
            return null;
        }
    }

    protected OutputLayer createOutputLayer() {
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
        if (!availableTypes.contains(m_pretrainedType) && m_pretrainedType != PretrainedType.NONE){
            log.error(String.format("%s weights are not available for this model, " +
                    "please try one of: %s", m_pretrainedType, availableTypes.toString()));
            return false;
        }
        return true;
    }

    private Set<PretrainedType> getAvailablePretrainedWeights(org.deeplearning4j.zoo.ZooModel zooModel) {
        Set<PretrainedType> availableTypes = new HashSet<>();
        for (PretrainedType pretrainedType : PretrainedType.values()) {
            if (pretrainedType == PretrainedType.NONE)
                continue;

            if (zooModel.pretrainedAvailable(pretrainedType.getBackend())) {
                availableTypes.add(pretrainedType);
            }
        }
        return availableTypes;
    }

    @OptionMetadata(
            description = "The name of the feature extraction layer in the model.",
            displayName = "Feature extraction layer",
            commandLineParamName = "layer-features",
            commandLineParamSynopsis = "-layer-features <String>"
    )
    public String getFeatureExtractionLayer() {
        return m_featureExtractionLayer;
    }

    public void setFeatureExtractionLayer(String featureExtractionLayer) {
        this.m_featureExtractionLayer = featureExtractionLayer;
    }

    @OptionMetadata(
            description = "The name of the output layer in the model.",
            displayName = "Output layer",
            commandLineParamName = "layer-output",
            commandLineParamSynopsis = "-layer-output <String>"
    )
    public String getOutputLayer() {
        return m_outputLayer;
    }

    public void setOutputLayer(String outputLayer) {
        this.m_outputLayer = outputLayer;
    }

    @OptionMetadata(
            description = "Number of feature extraction layer output nodes",
            displayName = "Num feature extract outputs",
            commandLineParamName = "layer-feature-outputs",
            commandLineParamSynopsis = "-layer-feature-outputs <int>"
    )
    public int getNumFExtractOutputs() {
        return m_numFExtractOutputs;
    }

    public void setNumFExtractOutputs(int m_numFExtractOutputs) {
        this.m_numFExtractOutputs = m_numFExtractOutputs;
    }

    @OptionMetadata(
            description = "Pretrained Type (IMAGENET, VGGFACE, IMAGENET LARGE, MNIST)",
            displayName = "Pretrained Type",
            commandLineParamName = "pretrained",
            commandLineParamSynopsis = "-pretrained <string>"
    )
    public PretrainedType getPretrainedType() {
        return m_pretrainedType;
    }

    public void setPretrainedType(PretrainedType pretrainedType) {
        setPretrainedType(pretrainedType, m_numFExtractOutputs, m_featureExtractionLayer, m_outputLayer, m_extraLayersToRemove);
    }

    protected AbstractZooModel setPretrainedType(PretrainedType pretrainedType,
                                                 int numFExtractOutputs,
                                                 String featureExtractionLayer,
                                                 String outputLayer) {
        return setPretrainedType(pretrainedType,
                numFExtractOutputs,
                featureExtractionLayer,
                outputLayer,
                new String[]{});
    }

    protected AbstractZooModel setPretrainedType(PretrainedType pretrainedType,
                                                 int numFExtractOutputs,
                                                 String featureExtractionLayer,
                                                 String outputLayer,
                                                 String[] extraLayersToRemove) {
        m_pretrainedType = pretrainedType;
        m_numFExtractOutputs = numFExtractOutputs;
        m_outputLayer = outputLayer;
        m_featureExtractionLayer = featureExtractionLayer;
        m_extraLayersToRemove = extraLayersToRemove;
        return this;
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
