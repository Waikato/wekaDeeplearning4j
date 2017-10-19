package weka.dl4j.listener;

import weka.core.Option;
import weka.core.OptionHandler;

import java.util.Enumeration;

public abstract class IterationListener implements org.deeplearning4j.optimize.api.IterationListener, OptionHandler{

    protected boolean invoked;

    protected int batchSize;
    protected int numEpochs;
    protected int numSamples;
    protected int numBatches;


    public void init(int numEpochs, int batchSize, int numSamples) {
        this.numEpochs = numEpochs;
        this.batchSize = batchSize;
        this.numSamples = numSamples;
        this.numBatches = numSamples/batchSize;
    }

    @Override
    public boolean invoked() {
        return invoked;
    }

    @Override
    public void invoke() {
        this.invoked = true;
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    @Override
    public Enumeration<Option> listOptions() {

        return Option.listOptionsForClass(this.getClass()).elements();
    }

    /**
     * Gets the current settings of the Classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    @Override
    public String[] getOptions() {

        return Option.getOptions(this, this.getClass());
    }

    /**
     * Parses a given list of options.
     *
     * @param options the list of options as an array of strings
     * @exception Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {

        Option.setOptions(options, this, this.getClass());
    }
}
