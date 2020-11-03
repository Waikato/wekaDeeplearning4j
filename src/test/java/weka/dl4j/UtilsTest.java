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

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import weka.dl4j.Utils;
import weka.core.Attribute;
import weka.core.Instances;
import weka.util.TestUtil;

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
}
