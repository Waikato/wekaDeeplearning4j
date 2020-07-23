package weka.dl4j.zoo;

import org.deeplearning4j.nn.graph.ComputationGraph;
import weka.core.OptionMetadata;
import weka.dl4j.PretrainedType;
import weka.dl4j.zoo.keras.VGG;
import weka.dl4j.zoo.keras.Xception;

public class KerasXception extends AbstractZooModel {

    private static final long serialVersionUID = -6899732453136761839L;

    private Xception.VARIATION variation = Xception.VARIATION.STANDARD;

    public KerasXception() {
        setVariation(Xception.VARIATION.STANDARD);
        setPretrainedType(PretrainedType.IMAGENET);
        setNumFExtractOutputs(2048);
        setFeatureExtractionLayer("avg_pool");
        setOutputLayer("predictions");
    }

    @OptionMetadata(
            description = "The model variation to use.",
            displayName = "Model Variation",
            commandLineParamName = "variation",
            commandLineParamSynopsis = "-variation <String>"
    )
    public Xception.VARIATION getVariation() {
        return variation;
    }

    /**
     * Does the model require input images to have the ImageNet preprocessing?
     *
     * @return true if the model input should be rescaled
     */
    @Override
    public boolean requiresImageNetScaling() {
        return true;
    }

    public void setVariation(Xception.VARIATION var) {
        variation = var;
    }

    @Override
    public ComputationGraph init(int numLabels, long seed, int[] shape, boolean filterMode) {
        Xception xception = new Xception();
        xception.setVariation(variation);

        return attemptToLoadWeights(xception, null, seed, numLabels, filterMode);
    }

    @Override
    public int[][] getShape() {
        int[][] shape = new int[1][];
        shape[0] = Xception.inputShape;
        return shape;
    }
}
