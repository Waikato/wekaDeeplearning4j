package weka.dl4j.zoo;

import org.deeplearning4j.nn.graph.ComputationGraph;
import weka.dl4j.PretrainedType;
import weka.dl4j.zoo.keras.DenseNet;
import weka.dl4j.zoo.keras.EfficientNet;

public class KerasEfficientNet extends AbstractZooModel {
//    private static final long serialVersionUID = -947378361661L;

    private EfficientNet.VARIATION variation = EfficientNet.VARIATION.EFFICIENTNET_B0;

    public KerasEfficientNet() {
        setVariation(EfficientNet.VARIATION.EFFICIENTNET_B0);
        setPretrainedType(PretrainedType.IMAGENET);
    }

    public EfficientNet.VARIATION getVariation() {
        return variation;
    }

    public void setVariation(EfficientNet.VARIATION var) {
        variation = var;
        // We may need to update the pretrained values based on the new variation
        setPretrainedType(m_pretrainedType);
    }

    @Override
    public void setPretrainedType(PretrainedType pretrainedType) {
        int numFExtractOutputs = -1;
        switch (variation) {
            case EFFICIENTNET_B0:
                numFExtractOutputs = 1280;
            case EFFICIENTNET_B1:
                numFExtractOutputs = 1280;
            case EFFICIENTNET_B2:
                numFExtractOutputs = 1408;
            case EFFICIENTNET_B3:
                numFExtractOutputs = 1536;
            case EFFICIENTNET_B4:
                numFExtractOutputs = 1792;
            case EFFICIENTNET_B5:
                numFExtractOutputs = 2048;
            case EFFICIENTNET_B6:
                numFExtractOutputs = 2304;
            case EFFICIENTNET_B7:
                numFExtractOutputs = 2560;
        }

        setPretrainedType(pretrainedType, numFExtractOutputs, "top_dropout", "probs");
    }

    @Override
    public ComputationGraph init(int numLabels, long seed, int[] shape) {
        EfficientNet efficientNet = new EfficientNet();
        efficientNet.setVariation(variation);
        ComputationGraph defaultNet = efficientNet.init();

        return attemptToLoadWeights(efficientNet, defaultNet, seed, numLabels);
    }

    @Override
    public int[][] getShape() {
        int[][] shape = new int[1][];
        shape[0] = EfficientNet.inputShape;
        return shape;
    }
}
