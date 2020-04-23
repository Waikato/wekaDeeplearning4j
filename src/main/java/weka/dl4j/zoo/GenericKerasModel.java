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
import weka.core.WekaPackageManager;
import weka.dl4j.PretrainedType;

import java.io.File;

/**
 *
 *
 * @author Rhys Compton
 */
public class GenericKerasModel extends AbstractZooModel {

  private static final long serialVersionUID = 7572785977498659081L;

  protected File kerasH5File = new File(WekaPackageManager.getPackageHome().toURI());

  @Override
  public ComputationGraph init(int numLabels, long seed, int[] shape, boolean filterMode) {
    try {
      ComputationGraph computationGraph = KerasModelImport.importKerasModelAndWeights(kerasH5File.getPath());

      return addFinalOutputLayer(computationGraph, seed, numLabels);
    } catch (Exception ex) {
      ex.printStackTrace();
      return null;
    }
  }

  @OptionMetadata(
          description = "The trained Keras model file.",
          displayName = "Keras H5 file",
          commandLineParamName = "model",
          commandLineParamSynopsis = "-model <File>",
          displayOrder = 1
  )
  public File getKerasH5File() {
    return kerasH5File;
  }

  public void setKerasH5File(File modelPath) {
    this.kerasH5File = modelPath;
  }

  @Override
  public int[][] getShape() {
    int[][] shape = new int[1][];
    shape[0] = new int[] {3, 224, 224};
    return shape;
  }
}
