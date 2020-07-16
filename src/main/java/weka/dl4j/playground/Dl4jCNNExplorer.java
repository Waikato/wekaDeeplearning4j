package weka.dl4j.playground;

import com.sun.jna.platform.win32.OaIdl;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.classifiers.functions.dl4j.Utils;
import weka.core.*;
import weka.dl4j.zoo.*;

import java.io.File;
import java.io.Serializable;
import java.util.Enumeration;

public class Dl4jImageModelPlayground implements Serializable, OptionHandler, RevisionHandler, CommandlineRunnable {

    /**
     * The classifier model this filter is based on.
     */
    protected File serializedModelFile = new File(WekaPackageManager.getPackageHome().toURI());

    /**
     * The zoo model to use, if we're not loading from the serialized model file
     */
    protected AbstractZooModel zooModelType;

    /**
     * Model used for feature extraction
     */
    protected Dl4jMlpClassifier model;

    private TopNPredictions currentPredictions;

    public void init() throws Exception {
        // TODO possibly refactor into makePrediction
        model = Utils.loadPlaygroundModel(serializedModelFile, zooModelType);
    }

    public void makePrediction(File imageFile) throws Exception {
        ModelOutputDecoder decoder = new ModelOutputDecoder(new ClassMap(ClassMap.BuiltInClassMap.IMAGENET));

        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        INDArray image = loader.asMatrix(imageFile);

        if (zooModelType.getChannelsLast())
            image = image.permute(0,2,3,1);

        INDArray dup = image.dup();
        INDArray result = model.outputSingle(dup);

        TopNPredictions[] predictions = decoder.decodePredictions(result);

        // Only processing a single image at the moment, not batch processing
        currentPredictions = predictions[0];

        String modelName; // TODO refactor into TopNPredictions (or Utils class)
        if (Utils.userSuppliedModelFile(serializedModelFile)) {
            modelName = "Custom trained Dl4jMlpClassifier";
        } else {
            modelName = zooModelType.getClass().getSimpleName() + " (" + zooModelType.getVariation() + ")";
        }

//        System.out.println(thisPrediction.toSummaryString(imageFile.getName(), modelName));
    }

    @OptionMetadata(
            displayName = "Serialized model file",
            description = "Pointer to file - saved Dl4jMlpClassifier"

    )
    public File getSerializedModelFile() {
        return serializedModelFile;
    }

    public void setSerializedModelFile(File serializedModelFile) {
        this.serializedModelFile = serializedModelFile;
    }

    public AbstractZooModel getZooModelType() {
        return zooModelType;
    }

    public void setZooModelType(AbstractZooModel zooModelType) {
        this.zooModelType = zooModelType;
    }

    public TopNPredictions getCurrentPredictions() {
        return currentPredictions;
    }

    public void setCurrentPredictions(TopNPredictions currentPredictions) {
        this.currentPredictions = currentPredictions;
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

    /**
     * Returns the revision string.
     *
     * @return the revision
     */
    @Override
    public String getRevision() {
        return null;
    }
}
