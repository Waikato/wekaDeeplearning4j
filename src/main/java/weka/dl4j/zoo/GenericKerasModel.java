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
 * CustomNet.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j.zoo;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.KerasSequentialModel;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelBuilder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import weka.core.OptionMetadata;
import weka.dl4j.PretrainedType;

/**
 *
 *
 * @author Rhys Compton
 */
public class GenericKerasModel extends AbstractZooModel {

  private static final long serialVersionUID = 549852142732L;

  // TODO change to file as in AbstractZooModel
  protected String kerasH5File = "";

  @Override
  public ComputationGraph init(int numLabels, long seed, int[] shape) {
    try {
      ComputationGraph computationGraph = KerasModelImport.importKerasModelAndWeights(kerasH5File);

      return addFinalOutputLayer(computationGraph, seed, numLabels);
    } catch (Exception ex) {
      ex.printStackTrace();
      return null;
    }
  }

  @OptionMetadata(
          displayName = "Keras Model File",
          description = "Location of the keras model weights"
  )
  public String getKerasH5File() {
    return kerasH5File;
  }

  public void setKerasH5File(String kerasH5File) {
    this.kerasH5File = kerasH5File;
  }

  @Override
  public int[][] getShape() {
    int[][] shape = new int[1][];
    shape[0] = new int[] {3, 224, 224};
    return shape;
  }
}
