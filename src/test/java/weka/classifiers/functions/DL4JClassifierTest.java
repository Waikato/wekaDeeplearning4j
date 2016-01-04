package weka.classifiers.functions;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class DL4JClassifierTest {
	
	public static void main(String[] args) throws Exception {
		
		DataSource ds = new DataSource("datasets/iris.arff");
		Instances data = ds.getDataSet();
		data.setClassIndex( data.numAttributes() - 1 );
		
		ChrisDL4JClassifier cls = new ChrisDL4JClassifier();
		cls.setLayers(new weka.dl4j.layers.Layer[] {
				new weka.dl4j.layers.DenseLayer(), new weka.dl4j.layers.DenseLayer() } );
		cls.buildClassifier(data);
		
		cls.distributionsForInstances(data);
		
	}

}
