
package weka.iterators.instance;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import lombok.extern.log4j.Log4j2;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import weka.core.Attribute;
import weka.core.Instances;
import weka.dl4j.iterators.instance.AbstractInstanceIterator;
import weka.dl4j.iterators.instance.sequence.RelationalInstanceIterator;
import weka.util.TestUtil;

/**
 * JUnit tests for the {@link RelationalInstanceIterator}
 *
 * @author Steven Lang
 */
@Log4j2
public class RelationalInstanceIteratorTest {

  /**
   * Seed
   */
  private static final int SEED = 42;
  private static Instances data;

  // Init test dataset
  static {
    try {
      data = TestUtil
          .makeTestDatasetRelational(SEED, 20, 2, Attribute.NOMINAL, 1, 2, 2, 2, 100);
      data.setClassIndex(data.numAttributes() - 1);
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  /**
   * Iterator object
   */
  private RelationalInstanceIterator rii;

  /**
   * Initialize iterator
   */
  @Before
  public void init() {
    this.rii = new RelationalInstanceIterator();
    this.rii.setTrainBatchSize(32);
  }

  /**
   * Test validate method with valid data
   *
   * @throws Exception Could not load data
   */
  @Test
  public void testValidateValidData() throws Exception {
    // Test valid setup
    this.rii.validate(data);
  }

  @Test
  public void testOutputFormat() throws Exception {
    for (int tl : Arrays.asList(10, 50, 200)) {
      rii.setTruncateLength(tl);
      for (int bs : Arrays.asList(1, 4, 8, 16)) {
        final DataSetIterator it = rii.getDataSetIterator(data, TestUtil.SEED, bs);
        assertEquals(bs, it.batch());
        assertEquals(Arrays.asList("0.0", "1.0"), it.getLabels());
        final DataSet next = it.next();

        // Check feature shape, expect: (batchsize x wordvecsize x sequencelength)
        final long[] shapeFeats = next.getFeatures().shape();
        final long[] expShapeFeats = {bs, 6, tl};
        assertEquals(expShapeFeats[0], shapeFeats[0]);
        assertEquals(expShapeFeats[1], shapeFeats[1]);
        assertTrue(expShapeFeats[2] >= shapeFeats[2]);

        // Check label shape, expect: (batchsize x numclasses x sequencelength)
        final long[] shapeLabels = next.getLabels().shape();
        final long[] expShapeLabels = {bs, data.numClasses(), tl};
        assertEquals(expShapeLabels[0], shapeLabels[0]);
        assertEquals(expShapeLabels[1], shapeLabels[1]);
        assertTrue(expShapeLabels[2] >= shapeLabels[2]);
      }
    }
  }


  /**
   * Test batch correct creation.
   *
   * @throws Exception IO error.
   */
  @Test
  public void testBatches() throws Exception {
    final int seed = 1;
    for (int batchSize : new int[]{1, 2, 5, 10}) {
      final int actual = countIterations(data, rii, seed, batchSize);
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
      it.next();
    }
    return count;
  }
}
