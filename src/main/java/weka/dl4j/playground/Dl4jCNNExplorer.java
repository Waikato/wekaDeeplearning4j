package weka.dl4j.playground;

import lombok.extern.log4j.Log4j2;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.classifiers.functions.dl4j.Utils;
import weka.core.*;
import weka.dl4j.preprocessors.ImageNetPreprocessor;
import weka.dl4j.zoo.*;

import java.io.File;
import java.io.Serializable;
import java.util.Enumeration;

/**
 * Tool to allow easy experimentation and exploration of a trained ComputationGraph - either from a previously trained
 * Dl4jMlpClassifier, or from a pretrained Zoo Model (default).
 * @author - Rhys Compton
 */
@Log4j2
public class Dl4jCNNExplorer implements Serializable, OptionHandler, CommandlineRunnable {

    /**
     * The classifier model this filter is based on.
     */
    protected File serializedModelFile = new File(Utils.defaultFileLocation());

    /**
     * The zoo model to use, if we're not loading from the serialized model file (default)
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

    /**
     * Predictions for the current image
     */
    protected TopNPredictions currentPredictions;

    /**
     * Initialize the ComputationGraph
     * @throws Exception Exceptions from loading the ComputationGraph
     */
    public void init() throws Exception {
        model = Utils.loadPlaygroundModel(serializedModelFile, zooModelType);
    }

    /**
     * Main entrypoint - Runs the loaded ComputationGraph on the supplied image, saving the predictions
     * @param imageFile Image to run prediction on
     * @throws Exception
     */
    public void makePrediction(File imageFile) throws Exception {
        // Load the image
        NativeImageLoader loader = new NativeImageLoader(224, 224, 3); // TODO take shape from loaded model
        INDArray image = loader.asMatrix(imageFile);

        // We may need to change the channel order if using a channelsLast model (e.g., EfficientNet)
        if (zooModelType.getChannelsLast()) {
            log.info("Permuting channel order of input image...");
            image = image.permute(0,2,3,1);
        }

        if (zooModelType.requiresImageNetScaling()) {
            log.info("Applying ImageNet scaling");
            ImageNetPreprocessor preprocessor = new ImageNetPreprocessor();
            DataSet dataSet = new org.nd4j.linalg.dataset.DataSet(image, Nd4j.zeros(0));
            preprocessor.preProcess(dataSet);
        }

        // Run prediction
        INDArray result = model.outputSingle(image.dup());

        // Decode and store the predictions
        currentPredictions = modelOutputDecoder.decodePredictions(result, imageFile.getName(), getModelName());
    }

    /**
     * Get the name of the loaded model
     * @return Model name
     */
    public String getModelName() {
        if (Utils.notDefaultFileLocation(serializedModelFile)) {
            return "Custom trained Dl4jMlpClassifier";
        } else {
            return zooModelType.getPrettyName();
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

    /**
     * Run this tool from the command line
     * @param toRun Object to run
     * @param options Command line options
     * @throws Exception
     */
    private void commandLineRun(Object toRun, String[] options) throws Exception {
        if (!(toRun instanceof Dl4jCNNExplorer)) {
            throw new IllegalArgumentException("Object to execute is not a "
                    + "Dl4jCNNExplorer!");
        }

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

        // Run the explorer
        explorer.init();
        explorer.makePrediction(new File(inputImagePath));
        // Output the results to the command line
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

    /**
     * Print the usage options to standard err
     */
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
        return Option.getOptionsForHierarchy(this, this.getClass());
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
