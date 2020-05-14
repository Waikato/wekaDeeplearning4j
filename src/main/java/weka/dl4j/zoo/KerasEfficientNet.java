package weka.dl4j.zoo;

import org.deeplearning4j.nn.graph.ComputationGraph;
import weka.core.OptionMetadata;
import weka.dl4j.PretrainedType;
import weka.dl4j.zoo.keras.DenseNet;
import weka.dl4j.zoo.keras.EfficientNet;


public class KerasEfficientNet {
    // Waiting on new release for swish activation function to be added
    // https://github.com/eclipse/deeplearning4j/pull/8801
    // Uncomment when DL4J upgraded to 1.0.0-beta7

//public class KerasEfficientNet extends AbstractZooModel {
//    private static final long serialVersionUID = -7261085237958094389L;
//
//    private EfficientNet.VARIATION variation = EfficientNet.VARIATION.EFFICIENTNET_B0;
//
//    public KerasEfficientNet() {
//        setVariation(EfficientNet.VARIATION.EFFICIENTNET_B0);
//        setPretrainedType(PretrainedType.IMAGENET);
//    }
//
//    @OptionMetadata(
//            description = "The model variation to use.",
//            displayName = "Model Variation",
//            commandLineParamName = "variation",
//            commandLineParamSynopsis = "-variation <String>"
//    )
//    public EfficientNet.VARIATION getVariation() {
//        return variation;
//    }
//
//    public void setVariation(EfficientNet.VARIATION var) {
//        variation = var;
//        // We may need to update the pretrained values based on the new variation
//        setPretrainedType(m_pretrainedType);
//    }
//
//    @Override
//    public void setPretrainedType(PretrainedType pretrainedType) {
//        int numFExtractOutputs = -1;
//        switch (variation) {
//            case EFFICIENTNET_B0:
//                numFExtractOutputs = 1280;
//                break;
//            case EFFICIENTNET_B1:
//                numFExtractOutputs = 1280;
//                break;
//            case EFFICIENTNET_B2:
//                numFExtractOutputs = 1408;
//                break;
//            case EFFICIENTNET_B3:
//                numFExtractOutputs = 1536;
//                break;
//            case EFFICIENTNET_B4:
//                numFExtractOutputs = 1792;
//                break;
//            case EFFICIENTNET_B5:
//                numFExtractOutputs = 2048;
//                break;
//            case EFFICIENTNET_B6:
//                numFExtractOutputs = 2304;
//                break;
//            case EFFICIENTNET_B7:
//                numFExtractOutputs = 2560;
//                break;
//        }
//
//        setPretrainedType(pretrainedType, numFExtractOutputs, "top_dropout", "probs");
//    }
//
//    @Override
//    public ComputationGraph init(int numLabels, long seed, int[] shape, boolean filterMode) {
//        EfficientNet efficientNet = new EfficientNet();
//        efficientNet.setVariation(variation);
//        ComputationGraph defaultNet = efficientNet.init();
//
//        return attemptToLoadWeights(efficientNet, defaultNet, seed, numLabels, filterMode);
//    }
//
//    @Override
//    public int[][] getShape() {
//        int[][] shape = new int[1][];
//        shape[0] = EfficientNet.inputShape;
//        return shape;
//    }
}
