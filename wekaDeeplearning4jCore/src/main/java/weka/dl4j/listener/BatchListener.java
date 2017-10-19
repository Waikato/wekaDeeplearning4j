package weka.dl4j.listener;

import org.deeplearning4j.nn.api.Model;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A listener that prints the model score every batch.
 * Inspired by ScoreIterationListener written by Adam Gibson.
 *
 * @author Steven Lang
 * @version $Revision: 1 $
 */
public class BatchListener extends IterationListener {
    private static final Logger log = LoggerFactory.getLogger(BatchListener.class);
    private long iterCount = 0;
    private int currentEpoch = 0;
    private int currentBatch = 0;


    @Override
    public boolean invoked() {
        return invoked;
    }

    @Override
    public void invoke() {
        this.invoked = true;
    }

    @Override
    public void iterationDone(Model model, int iteration) {
        if (numSamples <= 0)
            numSamples = 1;
        if (iterCount % numSamples == 0) {
            invoke();
            double result = model.score();
            log.info("Epoch [" + currentEpoch + "/" + numEpochs + "], Score: " + result);
            currentEpoch += 1;
        }

        if (iterCount % batchSize == 0) {
            invoke();
            double result = model.score();
            log.info("Iteration [" + currentBatch*batchSize + "/" + numSamples + "], Score: " + result);

            if (currentBatch == numBatches) {
                currentBatch = 0;
            } else {
                currentBatch += 1;
            }


        }

        iterCount++;
    }
}
