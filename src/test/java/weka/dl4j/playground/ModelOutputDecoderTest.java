package weka.dl4j.playground;

import junit.framework.TestCase;
import org.nd4j.linalg.api.ndarray.INDArray;
import weka.util.DatasetLoader;

public class ModelOutputDecoderTest extends TestCase {

    public void testDecodePredictions() throws Exception {
        INDArray modelPredictions = DatasetLoader.loadCarPredictions();

        ModelOutputDecoder decoder = new ModelOutputDecoder();
        decoder.setBuiltInClassMap(ModelOutputDecoder.ClassmapType.IMAGENET);

        TopNPredictions carPrediction = decoder.decodePredictions(modelPredictions);

        assertNotNull(carPrediction);

        Prediction highestProbPrediction = carPrediction.getTopPrediction();

        assertEquals(817, highestProbPrediction.getClassID());
        assertEquals("sports car, sport car", highestProbPrediction.getClassName());
        assertEquals(0.767, highestProbPrediction.getClassProbability(), 0.001);
    }
}