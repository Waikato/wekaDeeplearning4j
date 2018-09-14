/*
 * WekaDeeplearning4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * WekaDeeplearning4j is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with WekaDeeplearning4j.  If not, see <https://www.gnu.org/licenses/>.
 *
 * EarlyStoppingTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j.earlystopping;

import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TestName;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.Instances;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.util.DatasetLoader;
import weka.util.TestUtil;

import java.io.IOException;

/**
 * Test early stopping edge cases, such as:
 *
 * <p>1.1) maxEpochsNoImprovement > maxEpochs
 * <p>
 * 1.2) maxEpochsNoImprovement = maxEpochs
 * <p>
 * 1.3) maxEpochsNoImprovement = 0
 * <p>
 * 1.4) maxEpochsNoImprovement < 0
 * <p>2.1) validationSplit = 0
 * <p>
 * 2.2) validationSplit = 100 (expect exception)
 * <p>
 * 2.3) validationSplit = 105 (expect exception)
 * <p>
 * 2.4) validationSplit < 0
 *
 * <p>Expect no exceptions.
 *
 * @author Steven Lang
 */
public class EarlyStoppingTest {

    /**
     * Logger instance
     */
    private static final Logger logger = LoggerFactory.getLogger(EarlyStoppingTest.class);
    /**
     * Current name
     */
    @Rule
    public TestName name = new TestName();
    /**
     * Classifier
     */
    private Dl4jMlpClassifier clf;
    /**
     * Dataset mnist
     */
    private Instances dataMnist;
    /**
     * Mnist image loader
     */
    private ImageInstanceIterator idiMnist;
    /**
     * Start time for time measurement
     */
    private long startTime;

    @Before
    public void before() throws Exception {
        // Init mlp clf
        clf = new Dl4jMlpClassifier();
        clf.setSeed(TestUtil.SEED);
        clf.setNumEpochs(TestUtil.DEFAULT_NUM_EPOCHS);
        clf.setDebug(false);

        // Init data
        dataMnist = DatasetLoader.loadMiniMnistMeta();
        idiMnist = DatasetLoader.loadMiniMnistImageIterator();
        idiMnist.setTrainBatchSize(TestUtil.DEFAULT_BATCHSIZE);
        startTime = System.currentTimeMillis();
        clf.setInstanceIterator(idiMnist);
    }

    @After
    public void after() throws IOException {
        double time = (System.currentTimeMillis() - startTime) / 1000.0;
        logger.info("Testmethod: " + name.getMethodName());
        logger.info("Time: " + time + "s");
    }

    @Test
    public void testMaxEpochsNoImprovementBiggerThanMaxEpochs() throws Exception {
        testConfig(3, 5, 20);
    }

    @Test
    public void testMaxEpochsNoImprovementEqualToMaxEpochs() throws Exception {
        testConfig(5, 5, 20);
    }

    @Test
    public void testMaxEpochsNoImprovementZero() throws Exception {
        testConfig(5, 0, 20);
    }

    @Test(expected = RuntimeException.class)
    public void testMaxEpochsNoImprovementBelowZero() throws Exception {
        testConfig(5, -5, 20);
    }

    @Test
    public void validationSplitZero() throws Exception {
        testConfig(5, 3, 0);
    }

    @Test(expected = RuntimeException.class)
    public void validationSplitBelowZero() throws Exception {
        testConfig(5, 3, -5);
    }

    @Test(expected = RuntimeException.class)
    public void validationSplitOneHundred() throws Exception {
        testConfig(5, 3, 100);
    }

    @Test(expected = RuntimeException.class)
    public void validationSplitAboveOneHundred() throws Exception {
        testConfig(5, 3, 105);
    }

    /**
     * Test early stopping configuration.
     *
     * @param maxEpochs              Number of training epochs
     * @param maxEpochsNoImprovement Number of no improvement epochs
     * @param validationSplit        Validation split for the data
     * @throws Exception Something went wrong
     */
    public void testConfig(int maxEpochs, int maxEpochsNoImprovement, double validationSplit)
            throws Exception {
        EarlyStopping es = new EarlyStopping();
        es.setMaxEpochsNoImprovement(maxEpochsNoImprovement);
        es.setValidationSetPercentage(validationSplit);
        clf.setEarlyStopping(es);
        clf.setNumEpochs(maxEpochs);
        clf.buildClassifier(dataMnist);
    }
}
