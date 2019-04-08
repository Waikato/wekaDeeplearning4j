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
 * VGG19.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j.zoo;

import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import weka.dl4j.Preferences;

/**
 * A WEKA version of DeepLearning4j's VGGN19 ZooModel.
 *
 * @author Steven Lang
 */
public class VGG19 implements ZooModel {
  private static final long serialVersionUID = -4452023767749633607L;

  @Override
  public ComputationGraph init(int numLabels, long seed, int[] shape) {
    org.deeplearning4j.zoo.model.VGG19 net = org.deeplearning4j.zoo.model.VGG19.builder()
        .cacheMode(CacheMode.NONE)
        .workspaceMode(Preferences.WORKSPACE_MODE)
        .inputShape(shape)
        .numClasses(numLabels)
        .build();
    org.deeplearning4j.nn.conf.ComputationGraphConfiguration conf = net.conf();
    return new ComputationGraph(conf);
  }

  @Override
  public int[][] getShape() {
    return org.deeplearning4j.zoo.model.VGG19.builder().build().metaData().getInputShape();
  }
}
