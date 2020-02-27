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
 * VGG16.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j.zoo;

import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.zoo.PretrainedType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.dl4j.Preferences;

import java.io.IOException;

/**
 * A WEKA version of DeepLearning4j's VGGN16 ZooModel.
 *
 * @author Steven Lang
 */
public class VGG16 implements ZooModel {

  private static final long serialVersionUID = -6728816089752609851L;

  protected final Logger log = LoggerFactory.getLogger(VGG16.class);

  private static final String featureExtractionLayer = "fc2";

  PretrainedType m_pretrainedType;

  public VGG16() {}

  public VGG16(PretrainedType pretrainedType) {
    m_pretrainedType = pretrainedType;
  }

  @Override
  public ComputationGraph init(int numLabels, long seed, int[] shape) {
    org.deeplearning4j.zoo.model.VGG16 net = org.deeplearning4j.zoo.model.VGG16.builder()
        .cacheMode(CacheMode.NONE)
        .workspaceMode(Preferences.WORKSPACE_MODE)
        .inputShape(shape)
        .numClasses(numLabels)
        .build();
    org.deeplearning4j.nn.conf.ComputationGraphConfiguration conf = net.conf();

    // If no pretrained weights specified, simply return the standard model
    if (m_pretrainedType == null) {
      return new ComputationGraph(conf);
    }

    // If the specified pretrained weights aren't available, return the standard model
    if (!net.pretrainedAvailable(m_pretrainedType)) {
      log.error(String.format("%s weights are not available for this model, please try another type", m_pretrainedType));
      m_pretrainedType = null;
      return new ComputationGraph(conf);
    }

    // Else try load the weights into the model
    ComputationGraph vgg16;
    try {
      log.info(String.format("Downloading %s weights", m_pretrainedType));
      vgg16 = (ComputationGraph) net.initPretrained(m_pretrainedType);
      log.info(vgg16.summary());
    } catch (IOException ex) {
      ex.printStackTrace();
      return new ComputationGraph(conf);
    }

    //Construct a new model with the intended architecture and print summary
    ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
            .setFeatureExtractor(featureExtractionLayer) //the specified layer and below are "frozen"
            .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
            .addLayer("predictions",
                    new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .nIn(4096).nOut(numLabels)
                            .weightInit(new NormalDistribution(0, 0.2 * (2.0 / (4096 + numLabels)))) //This weight init dist gave better results than Xavier
                            .activation(Activation.SOFTMAX).build(),
                    featureExtractionLayer)
            .build();
    log.info(vgg16Transfer.summary());


    return vgg16Transfer;
  }

  @Override
  public int[][] getShape() {
    return org.deeplearning4j.zoo.model.VGG16.builder().build().metaData().getInputShape();
  }
}
