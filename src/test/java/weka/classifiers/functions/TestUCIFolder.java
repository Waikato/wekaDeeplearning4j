package weka.classifiers.functions;

import java.io.File;

import org.junit.Test;
import org.junit.Assert;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.dl4j.layers.Layer;

/**
 * Test all the UCI classification files
 * This is an expensive test, but a good
 * sanity check for the package
 * @author Christopher Beckham
 */
public class TestUCIFolder {

	public void trainSmallMlp(Instances data) throws Exception {
		Dl4jMlpClassifier cls = new Dl4jMlpClassifier();
		Layer[] layers = new Layer[] {
				new weka.dl4j.layers.DenseLayer(),
				new weka.dl4j.layers.OutputLayer()
		};
		cls.setLayers(layers);
		cls.setNumEpochs(1);
		cls.buildClassifier(data);
		double[][] preds = cls.distributionsForInstances(data);
		// ensure that the predictions are not nan
		for(int x = 0; x < preds.length; x++) {
			for(int y = 0; y < preds[x].length; y++) {
				Assert.assertEquals(true, Double.isFinite(preds[x][y]));
			}
		}
	}

	@Test
	public void test() throws Exception {

		File folder = new File("/Users/cjb60/Desktop/weka/datasets/UCI/");

		File[] files = folder.listFiles();
		for(File f : files) {
			if(f.getAbsolutePath().contains(".arff")) {
				DataSource ds = new DataSource(f.getAbsolutePath());
				Instances data = ds.getDataSet();
				data.setClassIndex( data.numAttributes() - 1 );
				trainSmallMlp(data);
			}
		}

	}

}
