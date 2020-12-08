package weka.dl4j.inference;

import junit.framework.TestCase;
import org.nd4j.linalg.api.ndarray.INDArray;
import weka.core.WekaException;
import weka.util.DatasetLoader;

import java.io.File;

/**
 * Test for the ModelOutputDecoder
 * @author - Rhys Compton
 */
public class ModelOutputDecoderTest extends TestCase {

    /**
     * Test that the decoder parses the predictions correctly
     * @throws Exception
     */
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
     * Test that the decoder can parse a class map from ARFF correctly
     */
    public void testClassmapFromArff() {
        // Arrange
        ModelOutputDecoder decoder = new ModelOutputDecoder();
        decoder.setBuiltInClassMap(ClassmapType.CUSTOM);
        decoder.setClassMapFile(new File("src/test/resources/nominal/mnist.meta.minimal.arff"));

        // Act
        String[] classmap = decoder.getClasses();

        // Assert
        assertEquals(10, classmap.length);
    }

    public void testClassmapFromCsv() {
        // Arrange
        ModelOutputDecoder decoder = new ModelOutputDecoder();
        decoder.setBuiltInClassMap(ClassmapType.CUSTOM);
        decoder.setClassMapFile(new File("src/test/resources/nominal/aptos_train.csv"));

        // Act
        String[] classmap = decoder.getClasses();

        // Assert
        assertEquals(5, classmap.length);
    }

    public void testClassmapFromImageNet() {
        // Arrange
        ModelOutputDecoder decoder = new ModelOutputDecoder();
        decoder.setBuiltInClassMap(ClassmapType.IMAGENET);

        // Act
        String[] classmap = decoder.getClasses();

        // Assert
        assertEquals(1000, classmap.length);
    }

    public void testClassmapFromDarknetImageNet() {
        // Arrange
        ModelOutputDecoder decoder = new ModelOutputDecoder();
        decoder.setBuiltInClassMap(ClassmapType.DARKNET_IMAGENET);

        // Act
        String[] classmap = decoder.getClasses();

        // Assert
        assertEquals(1000, classmap.length);
    }

    public void testClassmapFromVGGFace() {
        // Arrange
        ModelOutputDecoder decoder = new ModelOutputDecoder();
        decoder.setBuiltInClassMap(ClassmapType.VGGFACE);

        // Act
        String[] classmap = decoder.getClasses();

        // Assert
        assertEquals(2622, classmap.length);
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