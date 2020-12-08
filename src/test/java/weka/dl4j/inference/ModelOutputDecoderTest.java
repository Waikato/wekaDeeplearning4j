package weka.dl4j.inference;

import junit.framework.TestCase;
import org.nd4j.linalg.api.ndarray.INDArray;
import weka.core.WekaException;
import weka.util.DatasetLoader;

/**
 * Test for the ModelOutputDecoder
 * @author - Rhys Compton
 */
public class ModelOutputDecoderTest extends TestCase {

    public void testDecodePredictions() throws Exception {
        INDArray modelPredictions = DatasetLoader.loadCarPredictions();

        ModelOutputDecoder decoder = new ModelOutputDecoder();
        decoder.setBuiltInClassMap(ClassmapType.IMAGENET);

        TopNPredictions carPrediction = decoder.decodePredictions(modelPredictions);

        assertNotNull(carPrediction);

        Prediction highestProbPrediction = carPrediction.getTopPrediction();

        assertEquals(817, highestProbPrediction.getClassID());
        assertEquals("sports car, sport car", highestProbPrediction.getClassName());
        assertEquals(0.767, highestProbPrediction.getClassProbability(), 0.001);
    }

    /**
     * Tests that the method fails properly when an invalid setup is provided.
     * Will complain about number of prediction classes not matching size of class map
     * @throws Exception
     */
    public void testIncorrectSetup() throws Exception {
        INDArray modelPredictions = DatasetLoader.loadCarPredictions();

        ModelOutputDecoder decoder = new ModelOutputDecoder();
        decoder.setBuiltInClassMap(ClassmapType.VGGFACE);

        try {
            decoder.decodePredictions(modelPredictions);
        } catch (Exception ex) {
            assertTrue(ex instanceof WekaException);
            return;
        }
        fail();
    }
}