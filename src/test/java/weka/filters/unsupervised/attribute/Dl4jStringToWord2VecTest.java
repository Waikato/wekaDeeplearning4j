
package weka.filters.unsupervised.attribute;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;

/**
 * JUnit tests for the Dl4jStringToWord2Vec Filter.
 *
 * @author Steven Lang
 */
public class Dl4jStringToWord2VecTest {

  //    @Test
  public void testReuters() throws Exception {
    final String arffPath = "datasets/text/ReutersCorn-train.arff";
    ConverterUtils.DataSource ds = new ConverterUtils.DataSource(arffPath);
    final Instances data = ds.getDataSet();
    Dl4jStringToWord2Vec dl4jw2v = new Dl4jStringToWord2Vec();
    dl4jw2v.setInputFormat(data);
    Instances d = Filter.useFilter(data, dl4jw2v);
  }
}
