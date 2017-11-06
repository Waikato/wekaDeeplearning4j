package weka.dl4j.updater;

import org.nd4j.linalg.learning.config.IUpdater;
import weka.core.Option;
import weka.core.OptionHandler;

import java.util.Enumeration;

/**
 * Default Updater that implements WEKA option handling.
 *
 * @author Steven Lang
 */
public interface Updater extends OptionHandler, IUpdater {

    double DEFAULT_LEARNING_RATE = 0.1;

    /**
     * Constructor setting default learning rate to 0.1
     */
    default void setDefaults() {
        setLearningRate(DEFAULT_LEARNING_RATE);
    }

    /**
     * Get the learning rate
     *
     * @return Learning rate
     */
    double getLearningRate();

    /**
     * Set the learning rate
     *
     * @param learningRate Learning rate
     */
    void setLearningRate(double learningRate);

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    @Override
    default Enumeration<Option> listOptions() {

        return Option.listOptionsForClass(this.getClass()).elements();
    }

    /**
     * Gets the current settings of the Classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    @Override
    default String[] getOptions() {

        return Option.getOptions(this, this.getClass());
    }

    /**
     * Parses a given list of options.
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    default void setOptions(String[] options) throws Exception {

        Option.setOptions(options, this, this.getClass());
    }
}
