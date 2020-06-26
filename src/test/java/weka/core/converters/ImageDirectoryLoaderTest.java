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
 * ActivationELUTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.core.converters;

import lombok.extern.log4j.Log4j2;
import org.junit.Test;
import weka.core.Instances;
import weka.dl4j.ApiWrapperTest;
import weka.dl4j.activations.ActivationELU;
import weka.util.DatasetLoader;

import java.io.File;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Uses the plant-seedlings-small dataset as a test for the ImageDirectoryLoader
 * Loads the dataset (which should throw no errors), and then checks a few basic
 * properties of the new Instances object (num attributes, classes, instances, etc.)
 */
@Log4j2
public class ImageDirectoryLoaderTest {

  public static Instances loadPlantSeedlings() {
    ImageDirectoryLoader loader = new ImageDirectoryLoader();
    loader.setInputDirectory(new File(DatasetLoader.FILE_PATH_PLANT_SEEDLINGS));
    Instances inst = loader.getDataSet();
    inst.setClassIndex(1);
    return inst;
  }

  /**
   * Check that the IDL loads all classes from the plant-seedlings dataset
   */
  @Test
  public void testIDLNumClasses() {
    Instances plantLoaded = loadPlantSeedlings();
    assertEquals(plantLoaded.numClasses(), DatasetLoader.NUM_CLASSES_PLANT_SEEDLINGS);
  }

  /**
   * Check that the IDL loads all instances
   */
  @Test
  public void testIDLNumInstances() {
    Instances plantLoaded = loadPlantSeedlings();
    assertEquals(plantLoaded.numInstances(), DatasetLoader.NUM_INSTANCES_PLANT_SEEDLINGS);
  }

  /**
   * Check that the loaded instances have the correct number of attributes (2)
   */
  @Test
  public void testIDLNumAttributes() {
    Instances plantLoaded = loadPlantSeedlings();
    assertEquals(plantLoaded.numAttributes(), DatasetLoader.NUM_ATTRIBUTES_IMAGE_META);
  }

  /**
   * Check that the first attribute is a string (filepath) and the second is nominal (image classification)
   */
  @Test
  public void testIDLAttributes() {
    Instances plantLoaded = loadPlantSeedlings();
    assertTrue(plantLoaded.attribute(0).isString());
    assertTrue(plantLoaded.classAttribute().isNominal());
  }
}