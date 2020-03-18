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

import com.sun.tools.javac.jvm.Gen;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.zoo.PretrainedType;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * A dummy ZooModel which is empty.
 *
 * @author Steven Lang
 */
public class GenericKerasModel extends AbstractZooModel {

  private static final long serialVersionUID = 79654176542732L;

  @Override
  public ComputationGraph init(int numLabels, long seed, int[] shape) {
    try {
//      KerasModel kerasModel = new KerasModel().modelBuilder().modelJsonFilename(modelJsonPath)
//              .enforceTrainingConfig(false)
////              .inputShape(shape)
//              .weightsHdf5FilenameNoRoot(modelWeightsPath).enforceTrainingConfig(true).buildModel();
//      MultiLayerNetwork kerasModel = KerasModelImport.importKerasSequentialModelAndWeights(modelWeightsPath);
//      kerasModel.toComputationGraph();
      MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights("");
      NDArray inputArray = (NDArray) Nd4j.zeros(shape);
      model.setInput(inputArray);
      return model.toComputationGraph();
    } catch (Exception ex) {
      ex.printStackTrace();
      return null;
    }
  }

  @Override
  public int[][] getShape() {
    int[][] shape = new int[1][];
    shape[0] = new int[] {3, 224, 224};
    return shape;
  }
}
