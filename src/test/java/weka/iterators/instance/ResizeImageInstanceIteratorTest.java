package weka.iterators.instance;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import weka.core.Instance;
import weka.core.Instances;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.iterators.instance.ResizeImageInstanceIterator;
import weka.util.DatasetLoader;

import java.io.File;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * JUnit tests for the {@link ResizeImageInstanceIterator}
 *
 * @author Steven Lang
 */
public class ResizeImageInstanceIteratorTest {

  /** Seed */
  private static final int SEED = 42;
  /** Resized height */
  private static final int NEW_HEIGHT = 100;
  /** Resized width */
  private static final int NEW_WIDTH = 100;
  /** ResizeImageInstanceIterator object */
  private ResizeImageInstanceIterator rii;
  /** ImageInstanceIterator object */
  private ImageInstanceIterator iii;

  /** Initialize iterator */
  @Before
  public void init() {
    this.iii = new ImageInstanceIterator();
    this.iii.setImagesLocation(new File("datasets/nominal/mnist-minimal"));
    this.iii.setNumChannels(1);
    this.iii.setTrainBatchSize(1);
    this.iii.setWidth(28);
    this.iii.setHeight(28);
    this.rii = new ResizeImageInstanceIterator(iii, NEW_WIDTH, NEW_HEIGHT);
  }

  /** Test getDataSetIterator */
  @Test
  public void testGetIterator() throws Exception {
    final Instances metaData = DatasetLoader.loadMiniMnistMeta();
    final int batchSize = 1;
    final DataSetIterator it = this.rii.getDataSetIterator(metaData, SEED, batchSize);

    Set<Integer> labels = new HashSet<>();
    for (Instance inst : metaData) {
      int label = Integer.parseInt(inst.stringValue(1));
      final DataSet next = it.next();
      int itLabel = next.getLabels().argMax().getInt(0);
      Assert.assertEquals(label, itLabel);
      labels.add(label);

      long[] shape = next.getFeatures().shape();

      int batchIndex = 0;
      int channelIndex = 1;
      int heightIndex = 2;
      int widthIndex = 3;

      Assert.assertEquals(1, shape[batchIndex]);
      Assert.assertEquals(1, shape[channelIndex]);
      Assert.assertEquals(NEW_HEIGHT, shape[heightIndex]);
      Assert.assertEquals(NEW_WIDTH, shape[widthIndex]);
    }
    final List<Integer> collect =
        it.getLabels().stream().map(Integer::valueOf).collect(Collectors.toList());
    Assert.assertEquals(10, labels.size());
    Assert.assertTrue(labels.containsAll(collect));
    Assert.assertTrue(collect.containsAll(labels));
  }
}
