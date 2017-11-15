package weka.iterators.instance;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.core.Instance;
import weka.core.Instances;
import weka.dl4j.iterators.instance.ConvolutionInstanceIterator;
import weka.util.DatasetLoader;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * JUnit tests for the ConvolutionInstanceIterator {@link ConvolutionInstanceIterator}
 *
 * @author Steven Lang
 */
public class ConvolutionInstancesIteratorTest {

  /** Logger instance */
  private static final Logger logger =
      LoggerFactory.getLogger(ConvolutionInstancesIteratorTest.class);
  /** Seed */
  private static final int SEED = 42;
  /** Iterator object */
  private ConvolutionInstanceIterator cii;
  /** Data */
  private Instances mnistMiniArff;

  /** Initialize iterator */
  @Before
  public void init() throws Exception {
    this.cii = new ConvolutionInstanceIterator();
    this.cii.setNumChannels(1);
    this.cii.setTrainBatchSize(1);
    this.cii.setWidth(28);
    this.cii.setHeight(28);
    this.mnistMiniArff = DatasetLoader.loadMiniMnistArff();
  }

  /** Test getDataSetIterator */
  @Test
  public void testGetIterator() throws Exception {
    final int batchSize = 1;
    final DataSetIterator it = this.cii.getDataSetIterator(mnistMiniArff, SEED, batchSize);

    Set<Integer> labels = new HashSet<>();
    for (int i = 0; i < mnistMiniArff.size(); i++) {
      Instance inst = mnistMiniArff.get(i);
      int instLabel = Integer.parseInt(inst.stringValue(inst.numAttributes() - 1));
      final DataSet next = it.next();
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
