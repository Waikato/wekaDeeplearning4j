package weka.dl4j.zoo;

import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import weka.dl4j.Preferences;
import weka.dl4j.PretrainedType;
import weka.dl4j.zoo.keras.DenseNet;

public class KerasDenseNet extends AbstractZooModel {
//    private static final long serialVersionUID = -947378361661L;

    private DenseNet.VARIATION variation = DenseNet.VARIATION.DENSENET121;

    public KerasDenseNet() {
        setVariation(DenseNet.VARIATION.DENSENET121);
        setPretrainedType(PretrainedType.IMAGENET);
    }

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
            case DENSENET169:
                numFExtractOutputs = 1664;
            case DENSENET201:
                numFExtractOutputs = 1920;
        }

        setPretrainedType(pretrainedType, numFExtractOutputs, "avg_pool", "fc1000");
    }

    @Override
    public ComputationGraph init(int numLabels, long seed, int[] shape) {
        DenseNet denseNet = new DenseNet();
        denseNet.setVariation(variation);
        ComputationGraph defaultNet = denseNet.init();

        return attemptToLoadWeights(denseNet, defaultNet, seed, numLabels);
    }

    @Override
    public int[][] getShape() {
        int[][] shape = new int[1][];
        shape[0] = DenseNet.inputShape;
        return shape;
    }
}
