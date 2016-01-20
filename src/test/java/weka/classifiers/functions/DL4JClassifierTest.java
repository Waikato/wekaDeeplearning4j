package weka.classifiers.functions;

import org.junit.Ignore;
import org.junit.Test;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class DL4JClassifierTest {
	
	public Instances loadIris() throws Exception {
		DataSource ds = new DataSource("datasets/iris.arff");
		Instances data = ds.getDataSet();
		data.setClassIndex( data.numAttributes() - 1 );
		return data;
	}
	
	@Test
	public void denseTest() throws Exception {	
		Instances data = loadIris();
		Dl4jMlpClassifier cls = new Dl4jMlpClassifier();
		cls.setLayers(new weka.dl4j.layers.Layer[] {
				new weka.dl4j.layers.DenseLayer(),
				new weka.dl4j.layers.DenseLayer(),
				new weka.dl4j.layers.OutputLayer() 
		});
		cls.setTrainBatchSize(1);
		//cls.setDebugFile("/tmp/debug.txt");
		cls.buildClassifier(data);
		cls.distributionsForInstances(data);
	}
	
	@Ignore
	public void convTest() throws Exception {
		
		Instances data = loadIris();
		
		Dl4jMlpClassifier cls = new Dl4jMlpClassifier();
		cls.setLayers(new weka.dl4j.layers.Layer[] {
				new weka.dl4j.layers.Conv1DLayer(),
				new weka.dl4j.layers.OutputLayer() 
		});
		cls.setTrainBatchSize(1);
		cls.setDebugFile("/tmp/debug.txt");
		cls.buildClassifier(data);
		
		cls.distributionsForInstances(data);				
		
	}

}
