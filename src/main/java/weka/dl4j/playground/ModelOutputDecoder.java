package weka.dl4j.playground;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.core.RevisionHandler;
import weka.gui.ProgrammaticProperty;

import java.io.File;
import java.io.Serializable;
import java.util.Enumeration;

public class ModelOutputDecoder implements Serializable, OptionHandler {

    protected ClassMap classMap = new ClassMap();

    /**
     * Class Functionality
     */
    public TopNPredictions decodePredictions(INDArray predictions, String imageName, String modelName) throws Exception {
        // Get number of instances to predict for
        long[] shape = predictions.shape();

        if (shape.length == 1) {
            // Want arr to be [batch_size, numClasses], so reshape it to be like that
            predictions = reshapeSingleInstanceToBatch(predictions);
        }

        int numInstances = (int) shape[0];

        // Create the returning Prediction[]
        TopNPredictions[] result = new TopNPredictions[numInstances];

        // Decode each prediction
        for (int i = 0; i < numInstances; i++) {
            INDArray thisInstance = predictions.get(NDArrayIndex.point(i));

            TopNPredictions topNPredictions = new TopNPredictions(imageName, modelName);
            topNPredictions.process(thisInstance, classMap);

            result[i] = topNPredictions;
        }

        // Only supporting single images atm
        return result[0];
    }

    private INDArray reshapeSingleInstanceToBatch(INDArray array) {
        // TODO test this works
        long numClasses = array.shape()[0];
        return array.reshape(1, numClasses);
    }

    @OptionMetadata(
            displayName = "Class map",
            description = "Mapping from class IDs to human-readable class names",
            commandLineParamName = "classMap",
            commandLineParamSynopsis = "-classMap <options>"
    )
    public ClassMap getClassMap() {
        return classMap;
    }

    public void setClassMap(ClassMap classMap) {
        this.classMap = classMap;
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
