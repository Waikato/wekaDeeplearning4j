package weka.dl4j.listener;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;

/**
 * A listener that prints the model score every epoch.
 * Inspired by ScoreIterationListener written by Adam Gibson.
 *
 * @author Steven Lang
 */
public class EpochListener extends IterationListener implements TrainingListener {
    private static final Logger log = LoggerFactory.getLogger(weka.dl4j.listener.EpochListener.class);
    private static final long serialVersionUID = -8852994767947925554L;
    private int currentEpoch = 0;

    @Override
    public void onEpochEnd(Model model) {
        currentEpoch++;
        String s = "Epoch [" + currentEpoch + "/" + numEpochs + "]\n";
        s += "Train:      " + evaluateDataSetIterator(model, trainIterator);
        if (validationIterator != null){
            s += "Validation: " + evaluateDataSetIterator(model, validationIterator);
        }
        log(s);
        trainIterator.reset();
    }

    private String evaluateDataSetIterator(Model model, DataSetIterator iterator) {
        iterator.reset();
        String s = "";
        try {
            boolean isClassification = numClasses > 1;
            if (model instanceof ComputationGraph) {
                ComputationGraph net = (ComputationGraph) model;

                double scoreSum = 0;
                int iterations = 0;
                Evaluation cEval = new Evaluation(numClasses);
                RegressionEvaluation rEval = new RegressionEvaluation(1);
                while (iterator.hasNext()) {
                    DataSet next = iterator.next();
                    scoreSum += net.score(next);
                    iterations++;
                    INDArray output = net.outputSingle(next.getFeatureMatrix()); //get the networks prediction
                    if (isClassification) cEval.eval(next.getLabels(), output);
                    else rEval.eval(next.getLabels(), output);
                }

                double score = 0;
                if (iterations != 0){
                    score = scoreSum/iterations;
                }
                if (isClassification) {
                    s += String.format("Accuracy: %4.2f%%, ", cEval.accuracy() * 100);
                } else {
                    s += String.format("Avg R2: %4.2f, ", rEval.averagecorrelationR2());
                    s += String.format(", Avg RMSE: %4.2f, ", rEval.averagerootMeanSquaredError());
                }
                s += String.format("Loss: %9f\n", score);
            }
        } catch (UnsupportedOperationException e) {
            log.error("error: ", e);

        }


        return s;
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
