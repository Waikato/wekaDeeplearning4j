package weka.iterators.instance;


import org.datavec.image.recordreader.ImageRecordReader;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.util.DatasetLoader;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.InvalidObjectException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * JUnit tests for the ImageInstanceIterator {@link ImageInstanceIterator}
 *
 * @author Steven Lang
 */

public class ImageInstanceIteratorTest {

    /**
     * ImageInstanceIterator object
     */
    private ImageInstanceIterator idi;

    /**
     * Seed
     */
    private static final int SEED = 42;

    /**
     * Initialize iterator
     */
    @Before
    public void init() {
        this.idi = new ImageInstanceIterator();
        this.idi.setImagesLocation(new File("datasets/nominal/mnist-minimal"));
        this.idi.setNumChannels(1);
        this.idi.setTrainBatchSize(1);
        this.idi.setWidth(28);
        this.idi.setHeight(28);
    }

    /**
     * Test validate method with valid data
     *
     * @throws Exception Could not load mnist meta data
     */
    @Test
    public void testValidateValidData() throws Exception {
        // Test valid setup
        final Instances metaData = DatasetLoader.loadMiniMnistMeta();
        this.idi.validate(metaData);
    }

    /**
     * Test validate method with invalid data
     *
     * @throws Exception Could not load mnist meta data
     */
    @Test(expected = FileNotFoundException.class)
    public void testValidateInvalidLocation() throws Exception {
        final Instances metaData = DatasetLoader.loadMiniMnistMeta();
        final String invalidPath = "foo/bar/baz";
        this.idi.setImagesLocation(new File(invalidPath));
        this.idi.validate(metaData);
    }

    /**
     * Test validate method with invalid data
     *
     * @throws Exception Could not load mnist meta data
     */
    @Test(expected = InvalidObjectException.class)
    public void testValidateInvalidInstances() throws Exception {
        ArrayList<Attribute> invalidAttributes = new ArrayList<>();
        final Attribute f = new Attribute("file");
        invalidAttributes.add(f);
        final Instances metaData = new Instances("invalidMetaData", invalidAttributes, 0);
        this.idi.setImagesLocation(new File("datasets/nominal/mnist-minimal"));
        this.idi.validate(metaData);
    }


    /**
     * Test
     */
    @Test
    public void testGetImageRecordReader() throws Exception {
        final Instances metaData = DatasetLoader.loadMiniMnistMeta();
        Method method = ImageInstanceIterator.class.getDeclaredMethod("getImageRecordReader", Instances.class);
        method.setAccessible(true);
        this.idi.setTrainBatchSize(1);
        final ImageRecordReader irr = (ImageRecordReader) method.invoke(this.idi, metaData);

        Set<String> labels = new HashSet<>();
        for(Instance inst : metaData){
            String label = inst.stringValue(1);
            String itLabel = irr.next().get(1).toString();
            Assert.assertEquals(label, itLabel);
            labels.add(label);
        }
        Assert.assertEquals(10, labels.size());
        Assert.assertTrue(labels.containsAll(irr.getLabels()));
        Assert.assertTrue(irr.getLabels().containsAll(labels));
    }

    /**
     * Test getIterator
     */
    @Test
    public void testGetIterator() throws Exception {
        final Instances metaData = DatasetLoader.loadMiniMnistMeta();
        this.idi.setImagesLocation(new File("datasets/nominal/mnist-minimal"));
        final int batchSize = 1;
        final DataSetIterator it = this.idi.getIterator(metaData, SEED, batchSize);

        Set<Integer> labels = new HashSet<>();
        for(Instance inst : metaData){
            int label = Integer.parseInt(inst.stringValue(1));
            final DataSet next = it.next();
            int itLabel = next.getLabels().argMax().getInt(0);
            Assert.assertEquals(label, itLabel);
            labels.add(label);
        }
        final List<Integer> collect = it.getLabels().stream().map(Integer::valueOf).collect(Collectors.toList());
        Assert.assertEquals(10, labels.size());
        Assert.assertTrue(labels.containsAll(collect));
        Assert.assertTrue(collect.containsAll(labels));

    }
}
