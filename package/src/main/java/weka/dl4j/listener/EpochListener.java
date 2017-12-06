package weka.dl4j.listener;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import weka.core.OptionMetadata;

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

  /** Enable intermediate evaluations */
  private boolean enableIntermediateEvaluations = true;

  @Override
  public void onEpochEnd(Model model) {
    currentEpoch++;

    // Skip if this is not an evaluation epoch
    if (currentEpoch % n != 0) {
      return;
    }

    String s = "Epoch [" + currentEpoch + "/" + numEpochs + "]\n";

    if (enableIntermediateEvaluations) {
      s += "Train Set:      \n" + evaluateDataSetIterator(model, trainIterator);
      if (validationIterator != null) {
        s += "Validation Set: \n" + evaluateDataSetIterator(model, validationIterator);
      }
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

        Evaluation cEval = new Evaluation(numClasses);
        RegressionEvaluation rEval = new RegressionEvaluation(numClasses);
        while (iterator.hasNext()) {
          DataSet next;
          if (iterator instanceof AsyncDataSetIterator) {
            next = iterator.next();
          } else {
            // TODO: figure out which batch size is feasible for inference
            final int batch = iterator.batch() * 8;
            next = iterator.next(batch);
          }
          INDArray output =
              net.outputSingle(next.getFeatureMatrix()); // get the networks prediction
          if (isClassification) cEval.eval(next.getLabels(), output, next.getLabelsMaskArray());
          else rEval.eval(next.getLabels(), output, next.getLabelsMaskArray());
        }

        // Add loss
        s += String.format(" Loss:           %9f" + System.lineSeparator(), model.score());

        // Add Dl4j metrics
        if (isClassification) {
          final String stats =
              Arrays.stream(cEval.stats().split(System.lineSeparator()))
                  .filter(line -> !line.contains("# of classes")) // Remove # classes line
                  .filter(line -> !line.contains("===")) // Remove separators
                  .filter(line -> !line.contains("Examples labeled as")) // Remove confusion matrix
                  .filter(line -> !line.trim().isEmpty()) // Remove empty lines
                  .collect(Collectors.joining(System.lineSeparator())); // Join to original format
          s += stats + System.lineSeparator();
        } else {
          s += rEval.stats() + System.lineSeparator();
        }
      }
    } catch (UnsupportedOperationException e) {
      return "Set is too small and does not contain all labels." + System.lineSeparator();
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
      logFile.write(msg + System.lineSeparator());
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

  @OptionMetadata(
    displayName = "enable intermediate evaluations",
    description = "Enable intermediate evaluations (default = true).",
    commandLineParamName = "eval",
    commandLineParamSynopsis = "-eval <boolean>",
    displayOrder = 0
  )
  public boolean isEnableIntermediateEvaluations() {
    return enableIntermediateEvaluations;
  }

  public void setEnableIntermediateEvaluations(boolean enableIntermediateEvaluations) {
    this.enableIntermediateEvaluations = enableIntermediateEvaluations;
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
