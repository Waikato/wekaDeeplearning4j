package weka.dl4j.zoo;

import org.deeplearning4j.nn.graph.ComputationGraph;
import weka.dl4j.PretrainedType;
import weka.dl4j.zoo.keras.DenseNet;
import weka.dl4j.zoo.keras.MobileNet;

public class KerasMobileNet extends AbstractZooModel {
//    private static final long serialVersionUID = -947378361661L;

    private MobileNet.VARIATION variation = MobileNet.VARIATION.V2;

    public KerasMobileNet() {
        setVariation(MobileNet.VARIATION.V2);
        setPretrainedType(PretrainedType.IMAGENET);
    }

    public MobileNet.VARIATION getVariation() {
        return variation;
    }

    public void setVariation(MobileNet.VARIATION var) {
        variation = var;
        // We may need to update the pretrained values based on the new variation
        setPretrainedType(m_pretrainedType);
    }

    @Override
    public void setPretrainedType(PretrainedType pretrainedType) {
        switch (variation) {
            case V1:
                setPretrainedType(pretrainedType, 1000, "reshape_2", "act_softmax");
                break;
            case V2:
                setPretrainedType(pretrainedType, 1280, "global_average_pooling2d_1", "Logits");
                break;
        }
    }

    @Override
    public ComputationGraph init(int numLabels, long seed, int[] shape) {
        MobileNet mobileNet = new MobileNet();
        mobileNet.setVariation(variation);

        return attemptToLoadWeights(mobileNet, null, seed, numLabels);
    }

    @Override
    public int[][] getShape() {
        int[][] shape = new int[1][];
        shape[0] = MobileNet.inputShape;
        return shape;
    }
}
