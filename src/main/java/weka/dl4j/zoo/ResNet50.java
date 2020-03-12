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
 * ResNet50.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j.zoo;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.dl4j.Preferences;

import java.io.IOException;

/**
 * A WEKA version of DeepLearning4j's ResNet50 ZooModel.
 *
 * @author Steven Lang
 */
public class ResNet50 extends AbstractZooModel {

    private static final long serialVersionUID = -5206947378361661L;

    public ResNet50() {}

    @Override
    public ResNet50 setPretrainedType(PretrainedType pretrainedType) {
        return (ResNet50) setPretrainedType(pretrainedType, 2048, "fc1000", "flatten_1");
    }

    @Override
    public ComputationGraph init(int numLabels, long seed, int[] shape) {
        org.deeplearning4j.zoo.model.ResNet50 net = org.deeplearning4j.zoo.model.ResNet50.builder()
                .cacheMode(CacheMode.NONE)
                .workspaceMode(Preferences.WORKSPACE_MODE)
                .inputShape(shape)
                .numClasses(numLabels)
                .build();

        ComputationGraph defaultNet = net.init();

        return attemptToLoadWeights(net, defaultNet, seed, numLabels);
    }


    @Override
    public int[][] getShape() {
        return org.deeplearning4j.zoo.model.ResNet50.builder().build().metaData().getInputShape();
    }
}
