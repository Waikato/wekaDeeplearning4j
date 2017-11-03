package weka.dl4j.zoo;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/**
 * A WEKA version of DeepLearning4j's SimpleCNN ZooModel.
 *
 * @author Steven Lang
 */
public class SimpleCNN implements ZooModel {
    private static final long serialVersionUID = 4217466716595669736L;


    @Override
    public ComputationGraph init(int numLabels, long seed, int[][] shape) {
        org.deeplearning4j.zoo.model.SimpleCNN net = new org.deeplearning4j.zoo.model.SimpleCNN(numLabels, seed, 1);
        net.setInputShape(shape);
        org.deeplearning4j.nn.conf.MultiLayerConfiguration conf = net.conf();
        return mlpToCG(conf, shape);
    }

    @Override
    public int[][] getShape() {
        return new org.deeplearning4j.zoo.model.SimpleCNN().metaData().getInputShape();
    }
}
