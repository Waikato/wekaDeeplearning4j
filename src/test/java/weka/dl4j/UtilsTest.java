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
 * UtilTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.dl4j;

import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import weka.dl4j.Utils;
import weka.core.Attribute;
import weka.core.Instances;
import weka.dl4j.enums.PoolingType;
import weka.util.TestUtil;

import java.util.ArrayList;
import java.util.Collections;

public class UtilsTest {

  @Test
  public void testInstancesToDataSet() throws Exception {
    final Instances data = TestUtil.makeTestDataset(
        0,
        10,
        2,
        2,
        2,
        0,
        0,
        2,
        Attribute.NOMINAL,
        0,
        false
    );

    final DataSet dataSet = Utils.instancesToDataSet(data);
    final INDArray labels = dataSet.getLabels();
    final INDArray features = dataSet.getFeatures();

    for (int i = 0; i < data.numInstances(); i++) {

    }
  }

  @Test
  public void copyNominalAttribute_CopiesCorrectly() {
    // Arrange
    var attributeName = "Test Attribute";
    var attributeValues = new ArrayList<String>();
    attributeValues.add("Attribute val 1");
    attributeValues.add("Attribute val 2");
    attributeValues.add("Attribute val 3");
    var originalAttribute = new Attribute(attributeName, attributeValues);

    // Act
    var duplicateAttribute = Utils.copyNominalAttribute(originalAttribute);

    // Assert
    Assert.assertEquals(originalAttribute.numValues(), duplicateAttribute.numValues());

    var duplicateValues = Collections.list(duplicateAttribute.enumerateValues());
    Assert.assertArrayEquals(attributeValues.toArray(), duplicateValues.toArray());
  }

  @Test
  public void getAttributeName_Index0_IsLayer1() {
    // Arrange
    var attributesPerLayer = TestUtil.getAttributesPerLayer();
    var i = 0;

    // Act
    var layerName = Utils.getAttributeName(attributesPerLayer, i);

    Assert.assertEquals("layer1-0", layerName);
  }

  @Test
  public void getAttributeName_Index255_IsLayer1() {
    // Arrange
    var attributesPerLayer = TestUtil.getAttributesPerLayer();
    var i = 255;

    // Act
    var layerName = Utils.getAttributeName(attributesPerLayer, i);

    Assert.assertEquals("layer1-255", layerName);
  }

  @Test
  public void getAttributeName_Index256_IsLayer2() {
    // Arrange
    var attributesPerLayer = TestUtil.getAttributesPerLayer();
    var i = 256;

    // Act
    var layerName = Utils.getAttributeName(attributesPerLayer, i);

    Assert.assertEquals("layer2-0", layerName);
  }

  @Test
  public void getAttributeName_Index319_IsLayer2() {
    // Arrange
    var attributesPerLayer = TestUtil.getAttributesPerLayer();
    var i = 319;

    // Act
    var layerName = Utils.getAttributeName(attributesPerLayer, i);

    Assert.assertEquals("layer2-63", layerName);
  }

  @Test
  public void getAttributeName_OutofIndex_IsNull() {
    // Arrange
    var attributesPerLayer = TestUtil.getAttributesPerLayer();
    var i = 320;

    // Act
    var layerName = Utils.getAttributeName(attributesPerLayer, i);

    Assert.assertNull(layerName);
  }

  @Test
  public void getAttributeName_NoLayerMap_transformedAttribute() {
    // Arrange
    var i = 256;

    // Act
    var layerName = Utils.getAttributeName(null, i);

    Assert.assertEquals("transformedAttribute256", layerName);
  }

  @Test
  public void needsReshaping_2d_isFalse() {
    // Arrange
    var activations = Nd4j.rand(8, 64);

    // Assert
    Assert.assertFalse(Utils.needsReshaping(activations));
  }

  @Test
  public void needsReshaping_3d_isTrue() {
    // Arrange
    var activations = Nd4j.rand(8, 64, 7);

    // Assert
    Assert.assertTrue(Utils.needsReshaping(activations));
  }

  @Test
  public void needsReshaping_4d_isTrue() {
    // Arrange
    var activations = Nd4j.rand(8, 64, 7, 7);

    // Assert
    Assert.assertTrue(Utils.needsReshaping(activations));
  }

  @Test
  public void poolNDArray_Max_IsMax() {
    // Arrange
    var ndArray = TestUtil.get2DArray();

    // Assert
    Assert.assertEquals(5, (int) Utils.poolNDArray(ndArray, PoolingType.MAX));
  }

  @Test
  public void poolNDArray_Avg_IsAvg() {
    // Arrange
    var ndArray = TestUtil.get2DArray();

    // Assert
    Assert.assertEquals(1.125, Utils.poolNDArray(ndArray, PoolingType.AVG), 0.01);
  }

  @Test
  public void poolNDArray_Sum_IsSum() {
    // Arrange
    var ndArray = TestUtil.get2DArray();

    // Assert
    Assert.assertEquals(9, (int) Utils.poolNDArray(ndArray, PoolingType.SUM));
  }

  @Test
  public void poolNDArray_Min_IsMin() {
    // Arrange
    var ndArray = TestUtil.get2DArray();

    // Assert
    Assert.assertEquals(-2, (int) Utils.poolNDArray(ndArray, PoolingType.MIN));
  }

  @Test(expected = IllegalArgumentException.class)
  public void poolNDArray_PNorm_throwsException() {
    // Arrange
    var ndArray = TestUtil.get2DArray();

    // Assert
    Assert.assertEquals(-2, (int) Utils.poolNDArray(ndArray, PoolingType.PNORM));
  }

  @Test(expected = IllegalArgumentException.class)
  public void poolNDArray_None_throwsException() {
    // Arrange
    var ndArray = TestUtil.get2DArray();

    // Assert
    Assert.assertEquals(-2, (int) Utils.poolNDArray(ndArray, PoolingType.NONE));
  }

  @Test()
  public void isChannelsLast_TrueForChannelsLast() {
    // Arrange
    var ndArray = Nd4j.zeros(1, 56, 56, 128);

    // Assert
    Assert.assertTrue(Utils.isChannelsLast(ndArray));
  }

  @Test()
  public void isChannelsLast_FalseForChannelsFirst() {
    // Arrange
    var ndArray = Nd4j.zeros(1, 128, 56, 56);

    // Assert
    Assert.assertFalse(Utils.isChannelsLast(ndArray));
  }
}
