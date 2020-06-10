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
 * XCeption.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j.zoo;

import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import weka.dl4j.Preferences;
import weka.dl4j.PretrainedType;

/**
 * A WEKA version of DeepLearning4j's XCeption ZooModel.
 *
 * @author Steven Lang
 * @author Rhys Compton
 */
public class Dl4jXception extends AbstractZooModel {

  private static final long serialVersionUID = -4927205727389940364L;

  public Dl4jXception() {
    setPretrainedType(PretrainedType.IMAGENET);
  }

  @Override
  public void setPretrainedType(weka.dl4j.PretrainedType pretrainedType) {
    setPretrainedType(pretrainedType, 2048, "avg_pool", "predictions");
  }

  @Override
  public ComputationGraph init(int numLabels, long seed, int[] shape, boolean filterMode) {
    org.deeplearning4j.zoo.model.Xception net = org.deeplearning4j.zoo.model.Xception.builder()
        .cacheMode(CacheMode.NONE)
        .workspaceMode(Preferences.WORKSPACE_MODE)
        .inputShape(shape)
        .numClasses(numLabels)
        .build();

    ComputationGraph defaultNet = net.init();

    return attemptToLoadWeights(net, defaultNet, seed, numLabels, filterMode, true, false);
  }

  @Override
  public int[][] getShape() {
    return org.deeplearning4j.zoo.model.Xception.builder().build().metaData().getInputShape();
  }
}
