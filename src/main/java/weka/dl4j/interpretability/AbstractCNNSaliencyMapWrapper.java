package weka.dl4j.interpretability;

import org.deeplearning4j.nn.graph.ComputationGraph;
import weka.classifiers.functions.dl4j.Utils;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.dl4j.zoo.AbstractZooModel;

import java.io.File;
import java.io.Serializable;
import java.util.Enumeration;

// TODO Document

public abstract class AbstractCNNSaliencyMapWrapper implements Serializable, OptionHandler {

    protected int batchSize = 1;

    protected int targetClassID = -1;

    protected File outputFile = new File(Utils.defaultFileLocation());

    public abstract void process(File imageFile);

    public ComputationGraph getComputationGraph() {
        return computationGraph;
    }

    public void setComputationGraph(ComputationGraph computationGraph) {
        this.computationGraph = computationGraph;
    }

    protected ComputationGraph computationGraph;

    public AbstractZooModel getZooModel() {
        return zooModel;
    }

    public void setZooModel(AbstractZooModel zooModel) {
        this.zooModel = zooModel;
    }

    protected AbstractZooModel zooModel;

    @OptionMetadata(
            displayName = "Batch size",
            description = "The mini batch size to use for map generation",
            commandLineParamName = "bs",
            commandLineParamSynopsis = "-bs <int>",
            displayOrder = 1
    )
    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    @OptionMetadata(
            displayName = "Target Class",
            description = "Output class to generate saliency maps for; default is -1 (use the highest probability class)",
            commandLineParamName = "target-class",
            commandLineParamSynopsis = "-target-class <int>",
            displayOrder = 2
    )
    public int getTargetClassID() {
        return targetClassID;
    }

    public void setTargetClassID(int targetClassID) {
        this.targetClassID = targetClassID;
    }

    @OptionMetadata(
            displayName = "Output file location",
            description = "File for the saliency map to be saved in",
            commandLineParamName = "output",
            commandLineParamSynopsis = "-output <file location>",
            displayOrder = 3
    )
    public File getOutputFile() {
        return outputFile;
    }

    public void setOutputFile(File outputFileLocation) {
        this.outputFile = outputFileLocation;
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration<Option> listOptions() {
        return Option.listOptionsForClassHierarchy(this.getClass(), AbstractZooModel.class).elements();
    }

    /**
     * Gets the current settings of the Classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    public String[] getOptions() {
        return Option.getOptionsForHierarchy(this, AbstractZooModel.class);
    }

    /**
     * Parses a given list of options.
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {
        Option.setOptionsForHierarchy(options, this, AbstractZooModel.class);

        weka.core.Utils.checkForRemainingOptions(options);
    }

}
