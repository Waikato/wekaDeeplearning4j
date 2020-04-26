package weka.dl4j.zoo;

import org.deeplearning4j.nn.graph.ComputationGraph;
import weka.core.OptionMetadata;
import weka.dl4j.PretrainedType;
import weka.dl4j.zoo.keras.InceptionResNetV2;
import weka.dl4j.zoo.keras.InceptionV3;

public class KerasInceptionV3 extends AbstractZooModel {

    private static final long serialVersionUID = 4695067559456124034L;

    private InceptionV3.VARIATION variation = InceptionV3.VARIATION.STANDARD;

    public KerasInceptionV3() {
        setVariation(InceptionV3.VARIATION.STANDARD);
        setPretrainedType(PretrainedType.IMAGENET);
    }

    @OptionMetadata(
            description = "The model variation to use.",
            displayName = "Model Variation",
            commandLineParamName = "variation",
            commandLineParamSynopsis = "-variation <String>"
    )
    public InceptionV3.VARIATION getVariation() {
        return variation;
    }

    public void setVariation(InceptionV3.VARIATION var) {
        variation = var;
        // We may need to update the pretrained values based on the new variation
        setPretrainedType(m_pretrainedType);
    }

    @Override
    public void setPretrainedType(PretrainedType pretrainedType) {
        int numFExtractOutputs = -1;
        switch (variation) {
            case STANDARD:
                numFExtractOutputs = 2048;
                break;
        }

        setPretrainedType(pretrainedType, numFExtractOutputs, "avg_pool", "predictions");
    }

    @Override
    public ComputationGraph init(int numLabels, long seed, int[] shape, boolean filterMode) {
        InceptionV3 inceptionV3 = new InceptionV3();
        inceptionV3.setVariation(variation);

        return attemptToLoadWeights(inceptionV3, null, seed, numLabels, filterMode);
    }

    @Override
    public int[][] getShape() {
        int[][] shape = new int[1][];
        shape[0] = InceptionV3.inputShape;
        return shape;
    }
}
