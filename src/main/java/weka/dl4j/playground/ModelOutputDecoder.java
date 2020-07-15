package weka.dl4j.playground;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;

public class ModelOutputDecoder {

    private ClassMap classMap;

    /**
     * Constructors
     */

    public ModelOutputDecoder() {}

    public ModelOutputDecoder(ClassMap classMap) {
        this.classMap = classMap;
    }

    /**
     * Setters
     */
    public void setClassMap(ClassMap classMap) {
        this.classMap = classMap;
    }

    /**
     * Class Functionality
     */
    public TopNPredictions[] decodePredictions(INDArray predictions) throws Exception {
        // Get number of instances to predict for
        long[] shape = predictions.shape();

        if (shape.length == 1) {
            // Want arr to be [batch_size, numClasses], so reshape it to be like that
            predictions = reshapeSingleInstanceToBatch(predictions);
        }

        int numInstances = (int) shape[0];

        // Create the returning Prediction[]
        TopNPredictions[] result = new TopNPredictions[numInstances];

        int dimension = 0;
        // Decode each prediction
        for (int i = 0; i < numInstances; i++) {
            INDArray thisInstance = predictions.get(NDArrayIndex.point(i));

            TopNPredictions topNPredictions = new TopNPredictions();
            topNPredictions.process(thisInstance, classMap);

            result[i] = topNPredictions;
        }

        return result;
    }

    public TopNPredictions[] decodePredictions(double[][] predictions) throws Exception {
        return decodePredictions(Nd4j.create(predictions));
    }

    public TopNPredictions[] decodePredictions(double[] predictions) throws Exception {
        return decodePredictions(Nd4j.create(predictions));
    }

    private INDArray reshapeSingleInstanceToBatch(INDArray array) {
        // TODO test this works
        long numClasses = array.shape()[0];
        return array.reshape(1, numClasses);
    }
}
