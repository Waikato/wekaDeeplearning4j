package weka.dl4j.playground;

import lombok.extern.log4j.Log4j2;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.classifiers.functions.dl4j.Utils;
import weka.core.*;
import weka.core.converters.AbstractFileLoader;
import weka.core.converters.ImageDirectoryLoader;
import weka.dl4j.zoo.*;

import java.io.File;
import java.io.Serializable;
import java.util.Enumeration;

@Log4j2
public class Dl4jCNNExplorer implements Serializable, OptionHandler, CommandlineRunnable {

    /**
     * The classifier model this filter is based on.
     */
    protected File serializedModelFile = new File(Utils.defaultFileLocation());

    /**
     * The zoo model to use, if we're not loading from the serialized model file
     */
    protected AbstractZooModel zooModelType = new Dl4jResNet50();

    /**
     * Decodes the prediction IDs to human-readable format
     * Defaults to a decoder for IMAGENET classes
     */
    protected ModelOutputDecoder modelOutputDecoder = new ModelOutputDecoder();

    /**
     * Model used for feature extraction
     */
    protected Dl4jMlpClassifier model;

    protected TopNPredictions currentPredictions;

    public void init() throws Exception {
        // TODO possibly refactor into makePrediction
        model = Utils.loadPlaygroundModel(serializedModelFile, zooModelType);
    }

    public void makePrediction(File imageFile) throws Exception {
        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        INDArray image = loader.asMatrix(imageFile);

        if (zooModelType.getChannelsLast())
            image = image.permute(0,2,3,1);

        INDArray result = model.outputSingle(image.dup());

        // Only processing a single image at the moment, not batch processing
        currentPredictions = modelOutputDecoder.decodePredictions(result, imageFile.getName(), getModelName());
    }

    public String getModelName() {
        if (Utils.notDefaultFileLocation(serializedModelFile)) {
            return "Custom trained Dl4jMlpClassifier";
        } else {
            Enum variation = zooModelType.getVariation();
            if (variation == null) {
                return zooModelType.getClass().getSimpleName();
            } else {
                return zooModelType.getClass().getSimpleName() + " (" + variation + ")";
            }
        }
    }

    public TopNPredictions getCurrentPredictions() {
        return currentPredictions;
    }

    @OptionMetadata(
            displayName = "Serialized model file",
            description = "Pointer to file of saved Dl4jMlpClassifier",
            commandLineParamName = "modelFile",
            commandLineParamSynopsis = "-modelFile <file path>",
            displayOrder = 1
    )
    public File getSerializedModelFile() {
        return serializedModelFile;
    }

    public void setSerializedModelFile(File serializedModelFile) {
        this.serializedModelFile = serializedModelFile;
    }

    @OptionMetadata(
            displayName = "Pretrained zoo model",
            description = "Type of pretrained model to use for prediction (instead of trained Dl4jMlpClassifier)",
            commandLineParamName = "zooModel",
            commandLineParamSynopsis = "-zooModel <options>",
            displayOrder = 2
    )
    public AbstractZooModel getZooModelType() { // TODO figure out why not applying any non-default models
        return zooModelType;
    }

    public void setZooModelType(AbstractZooModel zooModelType) {
        this.zooModelType = zooModelType;
    }

    @OptionMetadata(
            displayName = "Model output decoder",
            description = "Handles decoding of the model predictions",
            commandLineParamName = "decoder",
            commandLineParamSynopsis = "-decoder <options>",
            displayOrder = 3
    )
    public ModelOutputDecoder getModelOutputDecoder() {
        return modelOutputDecoder;
    }

    public void setModelOutputDecoder(ModelOutputDecoder modelOutputDecoder) {
        this.modelOutputDecoder = modelOutputDecoder;
    }



    /**
     * Perform any setup stuff that might need to happen before execution.
     *
     * @throws Exception if a problem occurs during setup
     */
    @Override
    public void preExecution() throws Exception {

    }

    private void commandLineRun(Object toRun, String[] options) throws Exception {
        if (!(toRun instanceof Dl4jCNNExplorer)) {
            throw new IllegalArgumentException("Object to execute is not a "
                    + "Dl4jCNNExplorer!");
        }
        weka.core.WekaPackageManager.loadPackages(true);

        Dl4jCNNExplorer explorer = (Dl4jCNNExplorer) toRun;

        // Parse the command line options
        String inputImagePath;
        try {
            inputImagePath = weka.core.Utils.getOption("i", options);
            if (inputImagePath.equals("")) {
                throw new WekaException("Please supply an image file with the -i <image path> arg");
            }
            explorer.setOptions(options);
        } catch (Exception ex) {
            ex.printStackTrace();
            printInfo();
            return;
        }

        explorer.init();
        explorer.makePrediction(new File(inputImagePath));
        System.out.println(explorer.getCurrentPredictions().toSummaryString());
    }

    /**
     * Execute the supplied object.
     *
     * @param toRun   the object to execute
     * @param options any options to pass to the object
     * @throws Exception if a problem occurs.
     */
    @Override
    public void run(Object toRun, String[] options) throws Exception {
        ClassLoader origLoader = Thread.currentThread().getContextClassLoader();
        try {
            Thread.currentThread().setContextClassLoader(this.getClass().getClassLoader());

            commandLineRun(toRun, options);
        } finally {
            Thread.currentThread().setContextClassLoader(origLoader);
        }
    }

    private void printInfo() {
        System.err.println("\nUsage:\n" + "\tDl4jCNNExplorer [options]\n"
                + "\n" + "Options:\n");

        Enumeration<Option> enm =
                ((OptionHandler) new Dl4jCNNExplorer()).listOptions();
        while (enm.hasMoreElements()) {
            Option option = enm.nextElement();
            System.err.println(option.synopsis());
            System.err.println(option.description());
        }
    }

    /**
     * Perform any teardown stuff that might need to happen after execution.
     *
     * @throws Exception if a problem occurs during teardown
     */
    @Override
    public void postExecution() throws Exception {

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
     * @throws Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {
        Option.setOptionsForHierarchy(options, this, this.getClass());
        weka.core.Utils.checkForRemainingOptions(options);
    }
}
