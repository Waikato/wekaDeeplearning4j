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
 * SimpleCNN.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j.zoo;

import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.PretrainedType;
import weka.dl4j.Preferences;

/**
 * A WEKA version of DeepLearning4j's SimpleCNN ZooModel.
 *
 * @author Steven Lang
 */
public class SimpleCNN extends AbstractZooModel {

  private static final long serialVersionUID = 42184563995669736L; // TODO figure out why no output layers

  public SimpleCNN() { super(); }

  public SimpleCNN(PretrainedType pretrainedType) {
    // Note there aren't any pretrained weights currently available, values below are simply placeholders.
    super(pretrainedType, -1, "", "");
  }

  @Override
  public ComputationGraph init(int numLabels, long seed, int[] shape) {
    org.deeplearning4j.zoo.model.SimpleCNN net = org.deeplearning4j.zoo.model.SimpleCNN.builder()
        .cacheMode(CacheMode.NONE)
        .workspaceMode(Preferences.WORKSPACE_MODE)
        .inputShape(shape)
        .numClasses(numLabels)
        .build();
    org.deeplearning4j.nn.conf.MultiLayerConfiguration conf = net.conf();
    ComputationGraph defaultNet = mlpToCG(conf, shape);

    return attemptToLoadWeights(net, defaultNet, seed, numLabels);
  }

  @Override
  public int[][] getShape() {
    return org.deeplearning4j.zoo.model.SimpleCNN.builder().build().metaData().getInputShape();
  }
}
