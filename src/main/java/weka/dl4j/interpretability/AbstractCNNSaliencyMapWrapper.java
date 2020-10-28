package weka.dl4j.interpretability;

import lombok.extern.log4j.Log4j2;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang.StringUtils;
import org.deeplearning4j.nn.graph.ComputationGraph;
import weka.classifiers.functions.dl4j.Utils;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.core.Range;
import weka.core.progress.ProgressManager;
import weka.dl4j.zoo.AbstractZooModel;
import weka.gui.ProgrammaticProperty;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.Serializable;
import java.nio.Buffer;
import java.util.Arrays;
import java.util.Enumeration;

// TODO Document

@Log4j2
public abstract class AbstractCNNSaliencyMapWrapper implements Serializable, OptionHandler {

    /**
     * Displays progress of the current process (feature extraction, training, etc.)
     */
    protected ProgressManager progressManager;

    protected int batchSize = 1;

    protected int[] targetClassIDs = new int[] {-1};

    protected File outputFile = new File(Utils.defaultFileLocation());

    protected boolean normalizeHeatmap = true;

    protected String[] classMap;

    protected ComputationGraph computationGraph;

    protected AbstractZooModel zooModel;

    public abstract void processImage(File imageFile);

    public abstract BufferedImage generateHeatmapToImage();

    public void saveResult(BufferedImage completeCompositeImage) {
        if (Utils.notDefaultFileLocation(getOutputFile())) {
            log.info(String.format("Output file location = %s", getOutputFile()));
            try {
                ImageIO.write(completeCompositeImage, "png", getOutputFile());
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        } else {
            log.error("No output file location given - not saving saliency map");
        }
    }

    public String[] getClassMap() {
        return classMap;
    }

    public void setClassMap(String[] classMap) {
        this.classMap = classMap;
    }

    public ComputationGraph getComputationGraph() {
        return computationGraph;
    }

    public void setComputationGraph(ComputationGraph computationGraph) {
        this.computationGraph = computationGraph;
    }

    public AbstractZooModel getZooModel() {
        return zooModel;
    }

    public void setZooModel(AbstractZooModel zooModel) {
        this.zooModel = zooModel;
    }

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
            displayName = "Target Classes",
            description = "Output class to generate saliency maps for; default is -1 (use the highest probability class). " +
                    "This only needs to be set if wanting to use a non-default class from the *command line*; if using the *GUI*, " +
                    "the 'View Saliency Map' window contains the interface for setting this.",
            commandLineParamName = "target-classes",
            commandLineParamSynopsis = "-target-classes <int,int,...>",
            displayOrder = 2
    )
    @ProgrammaticProperty
    public String getTargetClassIDs() {
        return StringUtils.join(ArrayUtils.toObject(targetClassIDs), ",");
    }

    public void setTargetClassIDs(String targetClassIDs) {
        this.targetClassIDs = Arrays.stream(targetClassIDs.split(",")).mapToInt(Integer::parseInt).toArray();
    }

    public int[] getTargetClassIDsAsInt() {
        return targetClassIDs;
    }

    public void setTargetClassIDsAsInt(int[] targetClassIDs) {
        this.targetClassIDs = targetClassIDs;
    }


    @OptionMetadata(
            displayName = "Output file location",
            description = "File for the saliency map to be saved in",
            commandLineParamName = "output",
            commandLineParamSynopsis = "-output <file location>",
            displayOrder = 3
    )
    @ProgrammaticProperty
    public File getOutputFile() {
        return outputFile;
    }

    public void setOutputFile(File outputFileLocation) {
        this.outputFile = outputFileLocation;
    }

    @OptionMetadata(
            displayName = "Normalize heatmap",
            description = "When generating the heatmap, should the values be normalized to be in [0, 1]",
            commandLineParamName = "normalize",
            commandLineParamSynopsis = "-normalize",
            commandLineParamIsFlag = true
    )
    @ProgrammaticProperty
    public boolean getNormalizeHeatmap() {
        return normalizeHeatmap;
    }

    public void setNormalizeHeatmap(boolean normalizeHeatmap) {
        this.normalizeHeatmap = normalizeHeatmap;
    }

    public ProgressManager getProgressManager() {
        return progressManager;
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
