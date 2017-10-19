package weka;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import weka.core.Instance;
import weka.core.Instances;
import weka.dl4j.ArffPathLabelGenerator;
import weka.util.DatasetLoader;

import java.net.URI;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.HashSet;

/**
 * JUnit tests for the {@link ArffPathLabelGenerator}
 *
 * @author Steven Lang
 */

public class ArffPathLabelGeneratorTest {

    /**
     * Generator object
     */
    private ArffPathLabelGenerator gen;

    /**
     * MNIST metadata
     */
    private Instances metaData;

    /**
     * MNIST basepath
     */
    private String basePath;

    /**
     * Initialize generator.
     *
     * @throws Exception Loading mnist meta failed
     */
    @Before
    public void init() throws Exception {
        this.metaData = DatasetLoader.loadMiniMnistMeta();
        this.basePath = DatasetLoader.loadMiniMnistImageIterator().getImagesLocation().getAbsolutePath();
        this.gen = new ArffPathLabelGenerator(this.metaData, this.basePath);
    }

    /**
     * Test the getLabelForPath method.
     */
    @Test
    public void testGetLabelForPath() {
        for (Instance inst : this.metaData) {
            String path = Paths.get(this.basePath, inst.stringValue(0)).toString();
            String label = inst.stringValue(1);

            Assert.assertEquals(label, this.gen.getLabelForPath(path).toString());
            Assert.assertEquals(label, this.gen.getLabelForPath(URI.create(path)).toString());
        }
    }

    /**
     * Test the getPathUris method.
     */
    @Test
    public void testGetPathUris() {
        final Collection<URI> pathURIs = this.gen.getPathURIs();
        Collection<URI> metaDataUris = new HashSet<>();
        this.metaData.forEach(
                instance -> metaDataUris.add(Paths.get(this.basePath, instance.stringValue(0)).toUri())
        );
        Assert.assertTrue(metaDataUris.containsAll(pathURIs));
        Assert.assertTrue(pathURIs.containsAll(metaDataUris));
    }



}
