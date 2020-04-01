package weka.dl4j.zoo;

import org.deeplearning4j.nn.graph.ComputationGraph;
import weka.dl4j.PretrainedType;
import weka.dl4j.zoo.keras.MobileNet;
import weka.dl4j.zoo.keras.NASNet;

public class KerasNASNet extends AbstractZooModel {
//    private static final long serialVersionUID = -947378361661L;

    private NASNet.VARIATION variation = NASNet.VARIATION.MOBILE;

    public KerasNASNet() {
        setVariation(NASNet.VARIATION.MOBILE);
        setPretrainedType(PretrainedType.IMAGENET);
    }

    public NASNet.VARIATION getVariation() {
        return variation;
    }

    public void setVariation(NASNet.VARIATION var) {
        variation = var;
        // We may need to update the pretrained values based on the new variation
        setPretrainedType(m_pretrainedType);
    }

    @Override
    public void setPretrainedType(PretrainedType pretrainedType) {
        int numFExtractOutputs = -1;
        switch (variation) {
            case MOBILE:
                numFExtractOutputs = 1056;
                break;
            case LARGE:
                numFExtractOutputs = 4032;
                break;
        }

        setPretrainedType(pretrainedType, numFExtractOutputs, "global_average_pooling2d_1", "predictions");
    }

    @Override
    public ComputationGraph init(int numLabels, long seed, int[] shape) {
        NASNet nasNet = new NASNet();
        nasNet.setVariation(variation);

        return attemptToLoadWeights(nasNet, null, seed, numLabels);
    }

    @Override
    public int[][] getShape() {
        int[][] shape = new int[1][];
        shape[0] = NASNet.inputShape;
        return shape;
    }
}
