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
 * Darknet19.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j.zoo;

import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import weka.dl4j.Preferences;
import weka.dl4j.PretrainedType;
import weka.dl4j.zoo.keras.DenseNet;

/**
 * A WEKA version of DeepLearning4j's Darknet19 ZooModel.
 *
 * @author Steven Lang
 * @author Rhys Compton
 */
public class Darknet19 extends AbstractZooModel {

  private static final long serialVersionUID = -52066850554864161L;

  public enum VARIATION {INPUT224, INPUT448};

  protected VARIATION m_variation = VARIATION.INPUT224;

  public Darknet19() { setPretrainedType(PretrainedType.IMAGENET); }

  public void setVariation(VARIATION var) {
    m_variation = var;
  }

  @Override
  public void setPretrainedType(PretrainedType pretrainedType) {
    setPretrainedType(pretrainedType, 1000,
            "globalpooling", "softmax",
            new String[]{"loss"});
  }

  @Override
  public ComputationGraph init(int numLabels, long seed, int[] shape, boolean filterMode) {
    org.deeplearning4j.zoo.model.Darknet19 net = org.deeplearning4j.zoo.model.Darknet19.builder()
        .cacheMode(CacheMode.NONE)
        .workspaceMode(Preferences.WORKSPACE_MODE)
        .inputShape(shape)
        .numClasses(numLabels)
        .build();

    ComputationGraph defaultNet = net.init();

    return attemptToLoadWeights(net, defaultNet, seed, numLabels, filterMode);
  }

  @Override
  public int[][] getShape() {
    int imgSize = -1;
    if (m_variation == VARIATION.INPUT224) {
      imgSize = 224;
    } else if (m_variation == VARIATION.INPUT448) {
      imgSize = 448;
    }
    int[][] shape = new int[1][];
    shape[0] = new int[] {3, imgSize, imgSize};
    return shape;
  }
}
