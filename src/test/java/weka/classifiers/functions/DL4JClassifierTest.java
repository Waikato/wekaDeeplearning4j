package weka.classifiers.functions;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.nio.file.Files;
import java.util.List;

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
	
	public Dl4jMlpClassifier getMlp() {
		Dl4jMlpClassifier cls = new Dl4jMlpClassifier();
		cls.setLayers(new weka.dl4j.layers.Layer[] {
				new weka.dl4j.layers.DenseLayer(),
				new weka.dl4j.layers.DenseLayer(),
				new weka.dl4j.layers.OutputLayer() 
		});
		return cls;
	}
	
	/**
	 * Test a simple MLP on the Iris dataset. We're testing
	 * to see that none of the method calls like buildClassifer
	 * and distributionsForInstaneces are throwing exceptions.
	 * Furthermore, we expect the resulting debug file to have
	 * 10 entries (excluding the header), since we have set the
	 * number of iterations to be 10.
	 * @throws Exception
	 */
	@Test
	public void denseTest() throws Exception {	
		Instances data = loadIris();
		Dl4jMlpClassifier cls = getMlp();
		cls.setTrainBatchSize(50);
		cls.setNumIterations(10);
		String tmpFile = System.getProperty("java.io.tmpdir") + File.separator + "denseTest.txt";
		System.err.println("denseTest() tmp file: " + tmpFile);
		cls.setDebugFile(tmpFile);
		cls.buildClassifier(data);
		cls.distributionsForInstances(data);
		List<String> lines = Files.readAllLines(new File(tmpFile).toPath());
		assertEquals(lines.size(), 10+1);
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
