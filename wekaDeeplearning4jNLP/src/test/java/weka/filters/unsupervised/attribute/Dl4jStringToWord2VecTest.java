package weka.filters.unsupervised.attribute;

import org.junit.Test;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;

import java.io.File;
/**
 * JUnit tests for the Dl4jMlpClassifier.
 * Tests nominal classes with iris, numerical classes with diabetes and image
 * classification with minimal mnist.
 *
 * @author Steven Lang
 *
 * @version $Revision: 11711 $
 */
public class Dl4jStringToWord2VecTest {
    
    @Test
    public void testReuters() throws Exception {
        final String arffPath = "../datasets/text/ReutersCorn-train.arff";
        ConverterUtils.DataSource ds = new ConverterUtils.DataSource(arffPath);
        final Instances data = ds.getDataSet();
        Dl4jStringToWord2Vec dl4jw2v = new Dl4jStringToWord2Vec();
        dl4jw2v.setInputFormat(data);
        Instances d = Filter.useFilter(data, dl4jw2v);
    }
}
