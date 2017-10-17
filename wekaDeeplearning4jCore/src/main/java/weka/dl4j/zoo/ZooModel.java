package weka.dl4j.zoo;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import weka.core.Option;
import weka.core.OptionHandler;

import javax.naming.OperationNotSupportedException;
import java.io.Serializable;
import java.util.Enumeration;

public interface ZooModel extends Serializable, OptionHandler{
    /**
     * Initialize the ZooModel
     * @param numLabels Number of labels to adjust the output
     * @param seed Seed
     * @param shape Input shape to adjust the input
     * @return MultiLayerNetwork of the specified ZooModel
     * @throws OperationNotSupportedException Init(...) was not supported (only EmptyNet)
     */
    abstract public MultiLayerNetwork init(int numLabels, long seed, int[][] shape) throws OperationNotSupportedException;

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    @Override
    default public Enumeration<Option> listOptions() {

        return Option.listOptionsForClass(this.getClass()).elements();
    }

    /**
     * Gets the current settings of the Classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    @Override
    default public String[] getOptions() {

        return Option.getOptions(this, this.getClass());
    }

    /**
     * Parses a given list of options.
     *
     * @param options the list of options as an array of strings
     * @exception Exception if an option is not supported
     */
    default public void setOptions(String[] options) throws Exception {

        Option.setOptions(options, this, this.getClass());
    }
}
