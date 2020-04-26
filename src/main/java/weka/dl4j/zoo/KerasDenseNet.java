package weka.dl4j.zoo;

import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import weka.core.OptionMetadata;
import weka.dl4j.Preferences;
import weka.dl4j.PretrainedType;
import weka.dl4j.zoo.keras.DenseNet;

public class KerasDenseNet extends AbstractZooModel {

    private static final long serialVersionUID = 4978029404997176050L;

    private DenseNet.VARIATION variation = DenseNet.VARIATION.DENSENET121;

    public KerasDenseNet() {
        setVariation(DenseNet.VARIATION.DENSENET121);
        setPretrainedType(PretrainedType.IMAGENET);
    }

    @OptionMetadata(
            description = "The model variation to use.",
            displayName = "Model Variation",
            commandLineParamName = "variation",
            commandLineParamSynopsis = "-variation <String>"
    )
    public DenseNet.VARIATION getVariation() {
        return variation;
    }

    public void setVariation(DenseNet.VARIATION var) {
        variation = var;
        // We may need to update the pretrained values based on the new variation
        setPretrainedType(m_pretrainedType);
    }

    @Override
    public void setPretrainedType(PretrainedType pretrainedType) {
        int numFExtractOutputs = -1;
        switch (variation) {
            case DENSENET121:
                numFExtractOutputs = 1024;
                break;
            case DENSENET169:
                numFExtractOutputs = 1664;
                break;
            case DENSENET201:
                numFExtractOutputs = 1920;
                break;
        }

        setPretrainedType(pretrainedType, numFExtractOutputs, "avg_pool", "fc1000");
    }

    @Override
    public ComputationGraph init(int numLabels, long seed, int[] shape, boolean filterMode) {
        DenseNet denseNet = new DenseNet();
        denseNet.setVariation(variation);
        ComputationGraph defaultNet = denseNet.init();

        return attemptToLoadWeights(denseNet, defaultNet, seed, numLabels, filterMode);
    }

    @Override
    public int[][] getShape() {
        int[][] shape = new int[1][];
        shape[0] = DenseNet.inputShape;
        return shape;
    }
}
