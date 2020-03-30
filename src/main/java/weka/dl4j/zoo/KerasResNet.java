package weka.dl4j.zoo;

import org.deeplearning4j.nn.graph.ComputationGraph;
import weka.dl4j.PretrainedType;
import weka.dl4j.zoo.keras.DenseNet;
import weka.dl4j.zoo.keras.ResNet;

public class KerasResNet extends AbstractZooModel {
//    private static final long serialVersionUID = -947378361661L;

    // TODO try out messing with number of f extract outputs
    private ResNet.VARIATION variation = ResNet.VARIATION.RESNET50; // TODO refactor this  variation to just use Keras model variation

    public KerasResNet() {
        setVariation(ResNet.VARIATION.RESNET50);
        setPretrainedType(PretrainedType.IMAGENET);
    }

    public ResNet.VARIATION getVariation() {
        return variation;
    }

    public void setVariation(ResNet.VARIATION var) {
        variation = var;
        // We may need to update the pretrained values based on the new variation
        setPretrainedType(m_pretrainedType);
    }

    @Override
    public void setPretrainedType(PretrainedType pretrainedType) {
        setPretrainedType(pretrainedType, 2048, "avg_pool", "probs");
    }

    @Override
    public ComputationGraph init(int numLabels, long seed, int[] shape) {
        ResNet resNet = new ResNet();
        resNet.setVariation(variation);

        return attemptToLoadWeights(resNet, null, seed, numLabels);
    }

    @Override
    public int[][] getShape() {
        int[][] shape = new int[1][];
        shape[0] = ResNet.inputShape;
        return shape;
    }
}
