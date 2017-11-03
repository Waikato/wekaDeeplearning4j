package weka.dl4j.zoo;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import javax.naming.OperationNotSupportedException;
/**
 * A dummy ZooModel which is empty.
 *
 * @author Steven Lang
 */
public class EmptyNet implements ZooModel {
    private static final long serialVersionUID = 7131900848379752732L;

    @Override
    public ComputationGraph init(int numLabels, long seed, int[][] shape) throws OperationNotSupportedException {
        throw new OperationNotSupportedException("This model cannot be initialized as a MultiLayerNetwork.");
    }
    @Override
    public int[][] getShape() {
        return new int[0][0];
    }
}
