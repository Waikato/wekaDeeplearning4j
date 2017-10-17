package weka.dl4j.zoo;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/**
 * A WEKA version of DeepLearning4j's SimpleCNN ZooModel.
 *
 * @author Steven Lang
 * @version $Revision: 1 $
 */
public class SimpleCNN implements ZooModel {
    @Override
    public MultiLayerNetwork init(int numLabels, long seed, int[][] shape) {
        org.deeplearning4j.zoo.model.SimpleCNN net = new org.deeplearning4j.zoo.model.SimpleCNN(numLabels, seed, 1);
        net.setInputShape(shape);
        org.deeplearning4j.nn.conf.MultiLayerConfiguration conf = net.conf();
        return new MultiLayerNetwork(conf);
    }
}
