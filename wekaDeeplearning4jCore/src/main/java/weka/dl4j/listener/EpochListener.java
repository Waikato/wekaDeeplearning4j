package weka.dl4j.listener;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;

/**
 * A listener that prints the model score every epoch.
 * Inspired by ScoreIterationListener written by Adam Gibson.
 *
 * @author Steven Lang
 * @version $Revision: 1 $
 */
public class EpochListener extends IterationListener implements TrainingListener {
    private static final Logger log = LoggerFactory.getLogger(weka.dl4j.listener.EpochListener.class);
    private int currentEpoch = 0;




    @Override
    public void onEpochEnd(Model model) {
        currentEpoch++;
        final double score = model.score();
        log.info("Epoch [" + currentEpoch + "/" + numEpochs + "], Score: " + score);
    }

    @Override
    public void iterationDone(Model model, int iteration) {
    }

    @Override
    public void onEpochStart(Model model) {
    }

    @Override
    public void onForwardPass(Model model, List<INDArray> activations) {

    }

    @Override
    public void onForwardPass(Model model, Map<String, INDArray> activations) {

    }

    @Override
    public void onGradientCalculation(Model model) {

    }

    @Override
    public void onBackwardPass(Model model) {

    }
}
