package weka.dl4j.updater;

import org.nd4j.linalg.learning.config.IUpdater;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;

import java.util.Enumeration;

/**
 * Default Updater that implements WEKA option handling.
 *
 * @author Steven Lang
 */
public interface Updater extends OptionHandler, IUpdater{

    void setLearningRate(double learningRate);
    double getLearningRate();
    
    
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
