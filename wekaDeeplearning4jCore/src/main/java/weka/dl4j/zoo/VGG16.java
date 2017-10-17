package weka.dl4j.zoo;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/**
 * A WEKA version of DeepLearning4j's VGGN16 ZooModel.
 *
 * @author Steven Lang
 * @version $Revision: 1 $
 */
public class VGG16 implements ZooModel {
    @Override
    public MultiLayerNetwork init(int numLabels, long seed, int[][] shape) {
        org.deeplearning4j.zoo.model.VGG16 net = new org.deeplearning4j.zoo.model.VGG16(numLabels, seed, 1);
        net.setInputShape(shape);
        org.deeplearning4j.nn.conf.MultiLayerConfiguration conf = net.conf();
        return new MultiLayerNetwork(conf);
    }
}
