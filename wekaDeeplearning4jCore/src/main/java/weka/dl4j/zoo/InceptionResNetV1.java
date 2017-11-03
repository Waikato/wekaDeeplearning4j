package weka.dl4j.zoo;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A WEKA version of DeepLearning4j's InceptionResNetV1 ZooModel.
 *
 * @author Steven Lang
 */
public class InceptionResNetV1 implements ZooModel {
    private static final long serialVersionUID = -520668505548861661L;

    @Override
    public ComputationGraph init(int numLabels, long seed, int[][] shape) {
        org.deeplearning4j.zoo.model.InceptionResNetV1 net = new org.deeplearning4j.zoo.model.InceptionResNetV1(numLabels, seed, 1);
        net.setInputShape(shape);
        return net.init();
    }
    @Override
    public int[][] getShape() {
        return new org.deeplearning4j.zoo.model.InceptionResNetV1().metaData().getInputShape();
    }
}
