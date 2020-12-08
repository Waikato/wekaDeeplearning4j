package weka.dl4j.inference;

import junit.framework.TestCase;
import org.nd4j.linalg.api.ndarray.INDArray;
import weka.util.DatasetLoader;

public class TopNPredictionsTest extends TestCase {

    /**
     * Ensures that the predictions are correctly sorted in descending order
     */
    public void testPredictionsInCorrectOrder() throws Exception {
        INDArray modelPredictions = DatasetLoader.loadCarPredictions();

        ModelOutputDecoder decoder = new ModelOutputDecoder();
        decoder.setBuiltInClassMap(ClassmapType.IMAGENET);

        TopNPredictions carPredictions = decoder.decodePredictions(modelPredictions);

        assertNotNull(carPredictions);

        double lastPredictionProb = Double.MAX_VALUE;

        for (int i = 0; i < carPredictions.getN(); i++) {
            Prediction prediction = carPredictions.getPrediction(i);
            double thisPredictionProb = prediction.getClassProbability();
            assertTrue(lastPredictionProb > thisPredictionProb);
            lastPredictionProb = thisPredictionProb;
        }
    }
    
    
}