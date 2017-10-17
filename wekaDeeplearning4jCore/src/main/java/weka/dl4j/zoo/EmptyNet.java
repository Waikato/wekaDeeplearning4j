package weka.dl4j.zoo;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import javax.naming.OperationNotSupportedException;
/**
 * A dummy ZooModel which is empty.
 *
 * @author Steven Lang
 * @version $Revision: 1 $
 */
public class EmptyNet implements ZooModel {
    @Override
    public MultiLayerNetwork init(int numLabels, long seed, int[][] shape) throws OperationNotSupportedException {
        throw new OperationNotSupportedException("This model cannot be initialized");
    }
}
