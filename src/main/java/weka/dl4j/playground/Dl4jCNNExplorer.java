package weka.dl4j.playground;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.classifiers.functions.dl4j.Utils;
import weka.core.*;
import weka.dl4j.zoo.*;

import java.io.File;
import java.io.Serializable;
import java.util.Enumeration;

public class Dl4jCNNExplorer implements Serializable, OptionHandler, CommandlineRunnable {

    /**
     * The classifier model this filter is based on.
     */
    protected File serializedModelFile = new File(WekaPackageManager.getPackageHome().toURI());

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

    protected String getModelName() {
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

    /**
     * Execute the supplied object.
     *
     * @param toRun   the object to execute
     * @param options any options to pass to the object
     * @throws Exception if a problem occurs.
     */
    @Override
    public void run(Object toRun, String[] options) throws Exception {

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
        Option.setOptions(options, this, this.getClass());
    }
}
