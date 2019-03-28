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
 * ArffMetaDataLabelGeneratorTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka;

import java.io.File;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Collection;
import java.util.HashSet;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import weka.core.Instance;
import weka.core.Instances;
import weka.dl4j.ArffMetaDataLabelGenerator;
import weka.util.DatasetLoader;

/**
 * JUnit tests for the {@link ArffMetaDataLabelGenerator}
 *
 * @author Steven Lang
 */
public class ArffMetaDataLabelGeneratorTest {

  /**
   * Generator object
   */
  private ArffMetaDataLabelGenerator gen;

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
    this.basePath =
        DatasetLoader.loadMiniMnistImageIterator().getImagesLocation().getAbsolutePath();
    this.gen = new ArffMetaDataLabelGenerator(this.metaData, this.basePath);
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
      Assert.assertEquals(label, this.gen.getLabelForPath(new File(path).toURI()).toString());
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
        instance -> metaDataUris.add(Paths.get(this.basePath, instance.stringValue(0)).toUri()));
    Assert.assertTrue(metaDataUris.containsAll(pathURIs));
    Assert.assertTrue(pathURIs.containsAll(metaDataUris));
  }

  /**
   * Test the getPathUris method.
   */
  @Test
  public void testPathsWithSpaces() throws Exception {
    String originalArff = "src/test/resources/nominal/mnist.meta.minimal.arff";
    String originalDir = "src/test/resources/nominal/mnist-minimal";

    final String dir = "/tmp/nominal dir/";
    new File(dir).mkdir();
    String tmpArff = dir + "mnist.meta.minimal.arff";
    String tmpDir = dir + "mnist-minimal";
    Files.copy(Paths.get(originalArff), Paths.get(tmpArff), StandardCopyOption.REPLACE_EXISTING);
    Files.copy(Paths.get(originalDir), Paths.get(tmpDir), StandardCopyOption.REPLACE_EXISTING);

    this.metaData = DatasetLoader.loadArff(tmpArff);
    this.basePath =
        DatasetLoader.loadMnistImageIterator(tmpDir).getImagesLocation().getAbsolutePath();

    this.gen = new ArffMetaDataLabelGenerator(this.metaData, this.basePath);
    testGetLabelForPath();
    testGetPathUris();
  }
}
