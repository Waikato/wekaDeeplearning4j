package weka.dl4j.playground;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.classifiers.functions.dl4j.Utils;
import weka.core.*;
import weka.dl4j.zoo.*;

import java.io.File;

public class Dl4jModelExplorer {

    public File imageFile;
    /**
     * The classifier model this filter is based on.
     */
    protected File serializedModelFile = new File(WekaPackageManager.getPackageHome().toURI());

    /**
     * The zoo model to use, if we're not loading from the serialized model file
     */
    protected AbstractZooModel zooModelType = new Dl4jVGG();

    /**
     * Model used for feature extraction
     */
    protected Dl4jMlpClassifier model;

    public void init() throws Exception {
        // TODO possibly refactor into makePrediction
        model = Utils.loadPlaygroundModel(serializedModelFile, zooModelType);
    }

    public void makePrediction() throws Exception {
        ModelOutputDecoder decoder = new ModelOutputDecoder(new ClassMap(ClassMap.BuiltInClassMap.IMAGENET));

        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        INDArray image = loader.asMatrix(imageFile);

        INDArray result = model.outputSingle(image);

        Prediction[] predictions = decoder.decodePredictions(result);

        for (Prediction p : predictions) {
            System.out.println(p);
        }
    }

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
