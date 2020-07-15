package weka.dl4j.playground;

import junit.framework.TestCase;
import org.junit.Assert;
import org.nd4j.linalg.api.ndarray.INDArray;
import weka.util.DatasetLoader;

public class ModelOutputDecoderTest extends TestCase {

    public void testDecodePredictions() throws Exception {
        INDArray modelPredictions = DatasetLoader.loadCarPredictions();

        ModelOutputDecoder decoder = new ModelOutputDecoder(new ClassMap(ClassMap.BuiltInClassMap.IMAGENET));

        TopNPredictions[] decoded = decoder.decodePredictions(modelPredictions);

        assertNotNull(decoded);

        // Test there's only one in the batch
        assertEquals(decoded.length, 1);


        TopNPredictions carPrediction = decoded[0];

        // Get the highest probability prediction
        Prediction highestProbPrediction = carPrediction.getTopPrediction();

        assertEquals(817, highestProbPrediction.getClassID());

        assertEquals("sports car, sport car", highestProbPrediction.getClassName());

        assertEquals(0.767, highestProbPrediction.getClassProbability(), 0.001);
    }
}