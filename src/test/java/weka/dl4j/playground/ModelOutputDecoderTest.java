package weka.dl4j.playground;

import junit.framework.TestCase;
import org.junit.Assert;
import org.nd4j.linalg.api.ndarray.INDArray;
import weka.util.DatasetLoader;

public class ModelOutputDecoderTest extends TestCase {

    public void testDecodePredictions() throws Exception {
        INDArray modelPredictions = DatasetLoader.loadCarPredictions();

        ModelOutputDecoder decoder = new ModelOutputDecoder(new ClassMap(ClassMap.BuiltInClassMap.IMAGENET));

        Prediction[] decoded = decoder.decodePredictions(modelPredictions);

        assertNotNull(decoded);

        assertEquals(decoded.length, 1);

        Prediction carPrediction = decoded[0];

        assertEquals(817, carPrediction.getClassID());

        assertEquals("sports car, sport car", carPrediction.getClassName());

        assertEquals(0.767, carPrediction.getClassProbability(), 0.001);
    }
}