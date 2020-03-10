package weka.dl4j.zoo;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.zoo.PretrainedType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

public abstract class AbstractZooModel implements ZooModel {

    protected PretrainedType m_pretrainedType = null;

    private org.deeplearning4j.zoo.ZooModel m_zooModelType;

    protected final Logger log = LoggerFactory.getLogger(AbstractZooModel.class);

    protected String m_layerToRemove, m_featureExtractionLayer, m_predictionLayerName = "predictions";

    protected int m_numFExtractOutputs;

    public AbstractZooModel() {}

    public AbstractZooModel(PretrainedType pretrainedType,
                            int numFExtractOutputs,
                            String layerToRemove,
                            String featureExtractionLayer) {
        m_pretrainedType = pretrainedType;
        m_numFExtractOutputs = numFExtractOutputs;
        m_layerToRemove = layerToRemove;
        m_featureExtractionLayer = featureExtractionLayer;
    }

    public ComputationGraph attemptToLoadWeights(org.deeplearning4j.zoo.ZooModel zooModel,
                                                 ComputationGraph defaultNet,
                                                 long seed,
                                                 int numLabels) {
        // If no pretrained weights specified, simply return the standard model
        if (m_pretrainedType == null)
            return defaultNet;

        // If the specified pretrained weights aren't available, return the standard model
        if (!checkPretrained(zooModel)) {
            m_pretrainedType = null;
            return defaultNet;
        }

        // If downloading the weights fails, return the standard model
        ComputationGraph pretrainedModel = downloadWeights(zooModel);
        if (pretrainedModel == null)
            return defaultNet;

        // Finally, create the transfer learning graph
        ComputationGraph transferGraph;
        try {
            transferGraph = new TransferLearning.GraphBuilder(pretrainedModel)
                    .fineTuneConfiguration(getFineTuneConfig(seed))
                    .removeVertexKeepConnections(m_layerToRemove)
                    .addLayer(m_predictionLayerName, createOutputLayer(numLabels), m_featureExtractionLayer)
                    .setOutputs(m_predictionLayerName).build();
        } catch (Exception ex) {
            log.error("Couldn't load up weights for model");
            log.error(pretrainedModel.summary());
            return defaultNet;
        };

        return transferGraph;
    }

    protected FineTuneConfiguration getFineTuneConfig(long seed) {
        return new FineTuneConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(5e-5))
                .seed(seed)
                .build();
    }

    protected ComputationGraph downloadWeights(org.deeplearning4j.zoo.ZooModel net) {
        ComputationGraph cmpGraph;
        try {
            log.info(String.format("Downloading %s weights", m_pretrainedType));
            cmpGraph = (ComputationGraph) net.initPretrained(m_pretrainedType);
            return cmpGraph;
        } catch (IOException ex) {
            ex.printStackTrace();
            return null;
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
}
