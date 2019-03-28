
package weka.iterators.instance;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import weka.classifiers.functions.dl4j.Utils;
import weka.core.Attribute;
import weka.core.Instances;
import weka.util.TestUtil;

public class UtilTest {

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
