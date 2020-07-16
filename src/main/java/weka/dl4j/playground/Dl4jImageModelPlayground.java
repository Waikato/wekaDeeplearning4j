package weka.dl4j.playground;

import com.sun.jna.platform.win32.OaIdl;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.classifiers.functions.dl4j.Utils;
import weka.core.*;
import weka.dl4j.zoo.*;

import java.io.File;

public class Dl4jImageModelPlayground {

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
        TopNPredictions thisPrediction = predictions[0];

        String modelName; // TODO refactor into TopNPredictions (or Utils class)
        if (Utils.userSuppliedModelFile(serializedModelFile)) {
            modelName = "Custom trained Dl4jMlpClassifier";
        } else {
            modelName = zooModelType.getClass().getSimpleName() + " (" + zooModelType.getVariation() + ")";
        }

        System.out.println(thisPrediction.toSummaryString(imageFile.getName(), modelName));
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
}
