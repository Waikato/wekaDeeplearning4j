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
 * LeNet.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j.zoo;

import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.dl4j.Preferences;
import weka.dl4j.PretrainedType;

/**
 * A WEKA version of DeepLearning4j's LeNet ZooModel.
 *
 * @author Steven Lang
 * @author Rhys Compton
 */
public class Dl4jLeNet extends AbstractZooModel {

  private static final long serialVersionUID = 1656653625527283532L;

  public Dl4jLeNet() {
    setPretrainedType(PretrainedType.MNIST);
  }

  @Override
  public void setPretrainedType(PretrainedType pretrainedType) {
    setPretrainedType(pretrainedType,
            500,
            "7",
            "8",
            new String[] {"9"});
  }

  @Override
  public ComputationGraph init(int numLabels, long seed, int[] shape, boolean filterMode) {
    org.deeplearning4j.zoo.model.LeNet net = org.deeplearning4j.zoo.model.LeNet.builder()
        .cacheMode(CacheMode.NONE)
        .workspaceMode(Preferences.WORKSPACE_MODE)
        .inputShape(shape)
        .numClasses(numLabels)
        .build();

    ComputationGraph defaultNet = ((MultiLayerNetwork) net.init()).toComputationGraph();

    return attemptToLoadWeights(net, defaultNet, seed, numLabels, filterMode);
  }

  @Override
  public int[][] getShape() {
    return org.deeplearning4j.zoo.model.LeNet.builder().build().metaData().getInputShape();
  }
}
