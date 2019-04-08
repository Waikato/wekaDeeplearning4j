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
 * RnnSequenceClassifierAbstractTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.classifiers.functions;

import junit.framework.Test;
import junit.framework.TestSuite;
import weka.dl4j.layers.Layer;
import org.nd4j.linalg.lossfunctions.impl.LossSquaredHinge;
import weka.classifiers.AbstractClassifierTest;
import weka.classifiers.Classifier;
import weka.dl4j.earlystopping.EarlyStopping;
import weka.dl4j.layers.DenseLayer;
import weka.dl4j.layers.OutputLayer;
import weka.dl4j.layers.RnnOutputLayer;

/**
 * Abstract classifier test for the {@link RnnSequenceClassifier}.
 *
 * @author Steven Lang
 */
public class RnnSequenceClassifierAbstractTest extends AbstractClassifierTest {

    public RnnSequenceClassifierAbstractTest(String name) {
        super(name);
    }

    public static Test suite() {
        return new TestSuite(RnnSequenceClassifierAbstractTest.class);
    }

    public static void main(String[] args) {
        junit.textui.TestRunner.run(suite());
    }

    @Override
    public Classifier getClassifier() {
        RnnSequenceClassifier rnn = new RnnSequenceClassifier();
        RnnOutputLayer ol = new RnnOutputLayer();
        rnn.setLayers(ol);
        rnn.setNumEpochs(1);
        rnn.setEarlyStopping(new EarlyStopping(0, 0));
        return rnn;
    }
}
