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
 * ConvolutionInstancesIteratorTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.iterators.instance;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.functions.dl4j.Utils;
import weka.core.Instance;
import weka.core.Instances;
import weka.dl4j.iterators.instance.ConvolutionInstanceIterator;
import weka.util.DatasetLoader;

/**
 * JUnit tests for the ConvolutionInstanceIterator {@link ConvolutionInstanceIterator}
 *
 * @author Steven Lang
 */
public class ConvolutionInstancesIteratorTest {

  /**
   * Logger instance
   */
  private static final Logger logger =
      LoggerFactory.getLogger(ConvolutionInstancesIteratorTest.class);
  /**
   * Seed
   */
  private static final int SEED = 42;
  /**
   * Iterator object
   */
  private ConvolutionInstanceIterator cii;
  /**
   * Data
   */
  private Instances mnistMiniArff;

  /**
   * Initialize iterator
   */
  @Before
  public void init() throws Exception {
    this.cii = new ConvolutionInstanceIterator();
    this.cii.setNumChannels(1);
    this.cii.setTrainBatchSize(1);
    this.cii.setWidth(28);
    this.cii.setHeight(28);
    this.mnistMiniArff = DatasetLoader.loadMiniMnistArff();
  }

  /**
   * Test getDataSetIterator
   */
  @Test
  public void testGetIterator() throws Exception {
    final int batchSize = 1;
    final DataSetIterator it = this.cii.getDataSetIterator(mnistMiniArff, SEED, batchSize);

    Set<Integer> labels = new HashSet<>();
    for (int i = 0; i < mnistMiniArff.size(); i++) {
      Instance inst = mnistMiniArff.get(i);
      int instLabel = Integer.parseInt(inst.stringValue(inst.numAttributes() - 1));
      final DataSet next = Utils.getNext(it);
      int dsLabel = next.getLabels().argMax().getInt(0);
      Assert.assertEquals(instLabel, dsLabel);
      labels.add(instLabel);

      INDArray reshaped = next.getFeatures().reshape(1, inst.numAttributes() - 1);

      // Compare each attribute value
      for (int j = 0; j < inst.numAttributes() - 1; j++) {
        double instVal = inst.value(j);
        double dsVal = reshaped.getDouble(j);
        Assert.assertEquals(instVal, dsVal, 10e-8);
      }
    }

    final List<Integer> collect =
        it.getLabels().stream().map(Integer::valueOf).collect(Collectors.toList());
    Assert.assertEquals(10, labels.size());
    Assert.assertTrue(labels.containsAll(collect));
    Assert.assertTrue(collect.containsAll(labels));
  }
}
