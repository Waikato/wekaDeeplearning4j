package weka.dl4j.zoo;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import weka.core.OptionMetadata;
import weka.dl4j.enums.PretrainedType;
import weka.dl4j.zoo.keras.EfficientNet;

public class KerasEfficientNet extends AbstractZooModel {
    private static final long serialVersionUID = -7261085237958094389L;

    private EfficientNet.VARIATION variation = EfficientNet.VARIATION.EFFICIENTNET_B0;

    public KerasEfficientNet() {
        setVariation(EfficientNet.VARIATION.EFFICIENTNET_B0);
        setPretrainedType(PretrainedType.IMAGENET);
        setFeatureExtractionLayer("top_dropout");
        setOutputLayer("probs");
        setChannelsLast(true);
    }

    @OptionMetadata(
            description = "The model variation to use.",
            displayName = "Model Variation",
            commandLineParamName = "variation",
            commandLineParamSynopsis = "-variation <String>"
    )
    public EfficientNet.VARIATION getVariation() {
        return variation;
    }

    @Override
    public ImagePreProcessingScaler getImagePreprocessingScaler() {
        return new ImagePreProcessingScaler(-1, 1);
    }

    public void setVariation(EfficientNet.VARIATION var) {
        variation = var;
        int numFExtractOutputs = -1;
        switch (variation) {
            case EFFICIENTNET_B0:
                numFExtractOutputs = 1280;
                break;
            case EFFICIENTNET_B1:
                numFExtractOutputs = 1280;
                break;
            case EFFICIENTNET_B2:
                numFExtractOutputs = 1408;
                break;
            case EFFICIENTNET_B3:
                numFExtractOutputs = 1536;
                break;
            case EFFICIENTNET_B4:
                numFExtractOutputs = 1792;
                break;
            case EFFICIENTNET_B5:
                numFExtractOutputs = 2048;
                break;
            case EFFICIENTNET_B6:
                numFExtractOutputs = 2304;
                break;
            case EFFICIENTNET_B7:
                numFExtractOutputs = 2560;
                break;
        }
        setNumFExtractOutputs(numFExtractOutputs);
    }

    @Override
    public ComputationGraph init(int numLabels, long seed, int[] shape, boolean filterMode) {
        EfficientNet efficientNet = new EfficientNet();
        efficientNet.setVariation(variation);
        ComputationGraph defaultNet = efficientNet.init();

        return initZooModel(efficientNet, defaultNet, seed, numLabels, filterMode);
    }

    @Override
    public int[][] getShape() {
        int[][] shape = new int[1][];
        shape[0] = EfficientNet.inputShape;
        return shape;
    }
}
