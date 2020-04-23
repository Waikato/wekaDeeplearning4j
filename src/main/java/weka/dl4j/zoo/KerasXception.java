package weka.dl4j.zoo;

import org.deeplearning4j.nn.graph.ComputationGraph;
import weka.dl4j.PretrainedType;
import weka.dl4j.zoo.keras.VGG;
import weka.dl4j.zoo.keras.Xception;

public class KerasXception extends AbstractZooModel {

    private static final long serialVersionUID = -6899732453136761839L;
    
    private Xception.VARIATION variation = Xception.VARIATION.STANDARD;

    public KerasXception() {
        setVariation(Xception.VARIATION.STANDARD);
        setPretrainedType(PretrainedType.IMAGENET);
    }

    public Xception.VARIATION getVariation() {
        return variation;
    }

    public void setVariation(Xception.VARIATION var) {
        variation = var;
        // We may need to update the pretrained values based on the new variation
        setPretrainedType(m_pretrainedType);
    }

    @Override
    public void setPretrainedType(PretrainedType pretrainedType) {
        setPretrainedType(pretrainedType, 2048, "avg_pool", "predictions");
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
