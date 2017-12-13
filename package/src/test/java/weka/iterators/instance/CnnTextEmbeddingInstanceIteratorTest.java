package weka.iterators.instance;

import java.io.File;
import java.io.IOException;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.InvalidInputDataException;
import weka.dl4j.iterators.instance.AbstractInstanceIterator;
import weka.dl4j.iterators.instance.sequence.text.CnnTextEmbeddingInstanceIterator;
import weka.util.DatasetLoader;
import weka.util.TestUtil;

/**
 * JUnit tests for the {@link CnnTextEmbeddingInstanceIterator}
 *
 * @author Steven Lang
 */
@Slf4j
public class CnnTextEmbeddingInstanceIteratorTest {

  /** Seed */
  private static final int SEED = 42;
  /** Iterator object */
  private CnnTextEmbeddingInstanceIterator cteii;

  /** Initialize iterator */
  @Before
  public void init() throws IOException {
    this.cteii = new CnnTextEmbeddingInstanceIterator();
    this.cteii.setWordVectorLocation(DatasetLoader.loadGoogleNewsVectors());
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
    final Instances metaData = DatasetLoader.loadReutersFull();
    this.cteii.validate(metaData);
  }

  /**
   * Test validate method with invalid data
   *
   * @throws Exception Could not load data
   */
  @Test(expected = InvalidInputDataException.class)
  public void testValidateInvalidLocation() throws Exception {
    final Instances metaData = DatasetLoader.loadMiniMnistMeta();
    final String invalidPath = "foo/bar/baz";
    this.cteii.setWordVectorLocation(new File(invalidPath));
    this.cteii.validate(metaData);
  }

  /** Test getDataSetIterator */
  @Test
  public void testGetIteratorNominalClass() throws Exception {
    final Instances data = DatasetLoader.loadReutersMinimal();
    final int batchSize = 1;
    final DataSetIterator it = this.cteii.getDataSetIterator(data, SEED, batchSize);

    Set<Integer> labels = new HashSet<>();
    for (Instance inst : data) {
      int label = Integer.parseInt(inst.stringValue(data.classIndex()));
      final DataSet next = it.next();
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

    WordVectors wordVectors = WordVectorSerializer.loadStaticModel(DatasetLoader.loadGoogleNewsVectors());
    String[] words = (String[]) wordVectors.vocab().words().toArray(new String[0]);

    Random rand = new Random(42);
    for (Instance inst : data) {
      StringBuilder sentence = new StringBuilder();
      for(int i = 0; i < 10; i++){
        final int idx = rand.nextInt(words.length);
        sentence.append(" ").append(words[idx]);
      }
      inst.setValue(0, sentence.toString());
    }
    return data;
  }

  /** Test getDataSetIterator */
  @Test
  public void testGetIteratorNumericClass() throws Exception {
    final Instances data = makeData();
    final int batchSize = 1;
    final DataSetIterator it = this.cteii.getDataSetIterator(data, SEED, batchSize);

    Set<Double> labels = new HashSet<>();
    for (int i = 0; i < data.size(); i++) {
      Instance inst = data.get(i);
      double label = inst.value(data.classIndex());
      final DataSet next = it.next();
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
    Instances data = makeData();
    data.setClassIndex(data.numAttributes() - 1);

    final int seed = 1;
    for (int batchSize : new int[] {1, 2, 5, 10}) {
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
   * @throws Exception
   */
  private int countIterations(
      Instances data, AbstractInstanceIterator iter, int seed, int batchsize) throws Exception {
    DataSetIterator it = iter.getDataSetIterator(data, seed, batchsize);
    int count = 0;
    while (it.hasNext()) {
      count++;
      it.next();
    }
    return count;
  }
}
