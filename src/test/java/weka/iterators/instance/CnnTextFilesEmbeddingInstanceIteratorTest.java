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
 * CnnTextFilesEmbeddingInstanceIteratorTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.iterators.instance;

import java.io.File;
import java.io.IOException;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import lombok.extern.log4j.Log4j2;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import weka.classifiers.functions.dl4j.Utils;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.InvalidInputDataException;
import weka.dl4j.iterators.instance.AbstractInstanceIterator;
import weka.dl4j.iterators.instance.sequence.text.cnn.CnnTextFilesEmbeddingInstanceIterator;
import weka.util.DatasetLoader;
import weka.util.TestUtil;

/**
 * JUnit tests for the {@link CnnTextFilesEmbeddingInstanceIterator}
 *
 * @author Steven Lang
 */
@Log4j2
public class CnnTextFilesEmbeddingInstanceIteratorTest {

  /**
   * Seed
   */
  private static final int SEED = 42;
  /**
   * Iterator object
   */
  private CnnTextFilesEmbeddingInstanceIterator cteii;

  /**
   * Initialize iterator
   */
  @Before
  public void init() throws IOException {
    this.cteii = new CnnTextFilesEmbeddingInstanceIterator();
    this.cteii.setWordVectorLocation(DatasetLoader.loadGoogleNewsVectors());
    this.cteii.setTextsLocation(DatasetLoader.loadAngerFilesDir());
    this.cteii.setTrainBatchSize(32);
  }

  /**
   * Test validate method with valid data
   *
   * @throws Exception Could not load data
   */
  @Test
  public void testValidateValidData() throws Exception {
    // Test valid setup
    final Instances metaData = DatasetLoader.loadAngerMeta();
    this.cteii.validate(metaData);
  }

  /**
   * Test validate method with invalid data
   *
   * @throws Exception Could not load data
   */
  @Test(expected = InvalidInputDataException.class)
  public void testValidateInvalidLocation() throws Exception {
    final Instances metaData = DatasetLoader.loadAngerMeta();
    final String invalidPath = "foo/bar/baz";
    this.cteii.setWordVectorLocation(new File(invalidPath));
    this.cteii.validate(metaData);
  }

  /**
   * Test getDataSetIterator
   */
  @Test
  public void testGetIteratorNominalClass() throws Exception {
    final Instances data = DatasetLoader.loadAngerMetaClassification();
    final int batchSize = 1;
    final DataSetIterator it = this.cteii.getDataSetIterator(data, SEED, batchSize);

    Set<Integer> labels = new HashSet<>();
    for (int i = 0; i < data.size(); i++) {
      Instance inst = data.get(i);

      int label = Integer.parseInt(inst.stringValue(data.classIndex()));
      final DataSet next = Utils.getNext(it);
      int itLabel = next.getLabels().argMax().getInt(0);
      Assert.assertEquals(label, itLabel);
      labels.add(label);
    }
    final Set<Integer> collect =
        it.getLabels().stream().map(s -> Double.valueOf(s).intValue()).collect(Collectors.toSet());
    Assert.assertEquals(2, labels.size());
    Assert.assertTrue(labels.containsAll(collect));
    Assert.assertTrue(collect.containsAll(labels));
  }

  public Instances makeData() throws Exception {
    final Instances data = TestUtil.makeTestDataset(42,
        100,
        0,
        0,
        1,
        0,
        0,
        1,
        Attribute.NUMERIC,
        1,
        false);

    WordVectors wordVectors = WordVectorSerializer
        .loadStaticModel(DatasetLoader.loadGoogleNewsVectors());
    String[] words = (String[]) wordVectors.vocab().words().toArray(new String[0]);

    Random rand = new Random(42);
    for (Instance inst : data) {
      StringBuilder sentence = new StringBuilder();
      for (int i = 0; i < 10; i++) {
        final int idx = rand.nextInt(words.length);
        sentence.append(" ").append(words[idx]);
      }
      inst.setValue(0, sentence.toString());
    }
    return data;
  }

  /**
   * Test getDataSetIterator
   */
  @Test
  public void testGetIteratorNumericClass() throws Exception {
    final Instances data = DatasetLoader.loadAngerMeta();
    final int batchSize = 1;
    final DataSetIterator it = this.cteii.getDataSetIterator(data, SEED, batchSize);

    Set<Double> labels = new HashSet<>();
    for (int i = 0; i < data.size(); i++) {
      Instance inst = data.get(i);
      double label = inst.value(data.classIndex());
      final DataSet next = Utils.getNext(it);
      double itLabel = next.getLabels().getDouble(0);
      Assert.assertEquals(label, itLabel, 1e-5);
      labels.add(label);
    }
  }

  /**
   * Test batch correct creation.
   *
   * @throws Exception IO error.
   */
  @Test
  public void testBatches() throws Exception {

    // Data
    final Instances data = DatasetLoader.loadAngerMeta();

    final int seed = 1;
    for (int batchSize : new int[]{1, 2, 5, 10}) {
      final int actual = countIterations(data, cteii, seed, batchSize);
      final int expected = (int) Math.ceil(data.numInstances() / ((double) batchSize));
      Assert.assertEquals(expected, actual);
    }
  }

  /**
   * Counts the number of iterations
   *
   * @param data Instances to iterate
   * @param iter iterator to be tested
   * @param seed Seed
   * @param batchsize Size of the batch which is returned in {@see DataSetIterator#next}
   * @return Number of iterations
   */
  private int countIterations(
      Instances data, AbstractInstanceIterator iter, int seed, int batchsize) throws Exception {
    DataSetIterator it = iter.getDataSetIterator(data, seed, batchsize);
    int count = 0;
    while (it.hasNext()) {
      count++;
      Utils.getNext(it);
    }
    return count;
  }
}
