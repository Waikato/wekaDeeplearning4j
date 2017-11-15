package weka.dl4j.listener;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import weka.core.OptionMetadata;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;
import java.util.Map;

/**
 * A listener that prints the model score every epoch. Inspired by ScoreIterationListener written by
 * Adam Gibson.
 *
 * @author Steven Lang
 */
@Slf4j
public class EpochListener extends IterationListener implements TrainingListener {
  private static final long serialVersionUID = -8852994767947925554L;

  /** Epoch counter */
  private int currentEpoch = 0;

  /** Evaluate every N epochs */
  private int n = 5;

  /** Log to this file if set */
  private transient PrintWriter logFile;

  @Override
  public void onEpochEnd(Model model) {
    currentEpoch++;

    // Skip if this is not an evaluation epoch
    if (currentEpoch % n != 0) {
      return;
    }

    String s = "Epoch [" + currentEpoch + "/" + numEpochs + "]\n";
    s += "Train:      " + evaluateDataSetIterator(model, trainIterator);

    if (validationIterator != null) {
      s += "Validation: " + evaluateDataSetIterator(model, validationIterator);
    }

    log(s);
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
          // TODO: figure out which batch size is feasible for inference
          final int batch = iterator.batch() * 8;
          DataSet next = iterator.next(batch);
          scoreSum += net.score(next);
          iterations++;
          INDArray output =
              net.outputSingle(next.getFeatureMatrix()); // get the networks prediction
          if (isClassification) cEval.eval(next.getLabels(), output);
          else rEval.eval(next.getLabels(), output);
        }

        double score = 0;
        if (iterations != 0) {
          score = scoreSum / iterations;
        }
        if (isClassification) {
          s += String.format("Accuracy: %4.2f%%", cEval.accuracy() * 100);
        } else {
          s += String.format("Avg R2: %4.2f", rEval.averagecorrelationR2());
          s += String.format(", Avg RMSE: %4.2f", rEval.averagerootMeanSquaredError());
        }
        s += String.format(", Loss: %9f\n", score);
      }
    } catch (UnsupportedOperationException e) {
      return "Validation set is too small and does not contain all labels.";
    } catch (Exception e) {
      log.error("Evaluation after epoch failed. Error: ", e);
      return "Not available";
    } finally {
      iterator.reset();
    }

    return s;
  }

  /**
   * Set the log file
   *
   * @param logFile Logging file
   */
  public void setLogFile(File logFile) throws IOException {
    if (logFile.exists()) logFile.delete();
    this.logFile = new PrintWriter(new FileWriter(logFile, false));
  }

  @Override
  public void log(String msg) {
    log.info(msg);
    if (logFile != null) {
      logFile.write(msg + "\n");
      logFile.flush();
    }
  }

  @Override
  public void iterationDone(Model model, int iteration) {}

  @Override
  public void onEpochStart(Model model) {}

  @Override
  public void onForwardPass(Model model, List<INDArray> activations) {}

  @Override
  public void onForwardPass(Model model, Map<String, INDArray> activations) {}

  @Override
  public void onGradientCalculation(Model model) {}

  @Override
  public void onBackwardPass(Model model) {}

  public int getN() {
    return this.n;
  }

  @OptionMetadata(
    displayName = "evaluate every N epochs",
    description = "Evaluate every N epochs (default = 5).",
    commandLineParamName = "n",
    commandLineParamSynopsis = "-n <int>",
    displayOrder = 0
  )
  public void setN(int evaluateEveryNEpochs) {
    if (evaluateEveryNEpochs < 1) {
      // Never evaluate
      this.n = Integer.MAX_VALUE;
    } else {
      this.n = evaluateEveryNEpochs;
    }
  }

  /**
   * Returns a string describing this search method
   *
   * @return a description of the search method suitable for displaying in the explorer/experimenter
   *     gui
   */
  public String globalInfo() {
    return "A listener which evaluates the model while training every N " + "epochs.";
  }
}
