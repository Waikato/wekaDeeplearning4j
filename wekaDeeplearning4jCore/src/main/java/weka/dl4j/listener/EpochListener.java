package weka.dl4j.listener;

import lombok.Builder;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.core.logging.OutputLogger;

import java.io.BufferedOutputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

/**
 * A listener that prints the model score every epoch.
 * Inspired by ScoreIterationListener written by Adam Gibson.
 *
 * @author Steven Lang
 * @version $Revision: 1 $
 */
@Builder
public class EpochListener extends IterationListener implements TrainingListener {
    private static final Logger log = LoggerFactory.getLogger(weka.dl4j.listener.EpochListener.class);
    private static final long serialVersionUID = -8852994767947925554L;
    private int currentEpoch = 0;

    @Override
    public void onEpochEnd(Model model) {
        currentEpoch++;
        String s = "Epoch [" + currentEpoch + "/" + numEpochs + "]";
        int numClasses = iterator.totalOutcomes();
        boolean isClassification = numClasses > 1;
        if (model instanceof MultiLayerNetwork){
            MultiLayerNetwork net = (MultiLayerNetwork) model;

            Evaluation cEval = new Evaluation(numClasses);
            RegressionEvaluation rEval = new RegressionEvaluation(1); // Currently no multitarget
            while (iterator.hasNext()) {
                DataSet next = iterator.next();
                INDArray output = net.output(next.getFeatureMatrix()); //get the networks prediction
                if (isClassification) cEval.eval(next.getLabels(), output);
                else rEval.eval(next.getLabels(), output);
            }
            iterator.reset();
            if (isClassification){
                s += String.format(", Accuracy: %4.2f%%", cEval.accuracy()*100);
            } else {
                s += String.format(", Avg R2: %4.2f", rEval.averagecorrelationR2());
                s += String.format(", Avg RMSE: %4.2f", rEval.averagerootMeanSquaredError());
            }
        }

        s += String.format(", Loss: %9f", model.score());
        OutputLogger.log(weka.core.logging.Logger.Level.ALL, s);
        log(s);
    }

    @Override
    public void log(String msg) {
        log.info(msg);
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
