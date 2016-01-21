package weka.classifiers.functions;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;

import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import weka.classifiers.functions.dl4j.Utils;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.dl4j.Activation;
import weka.dl4j.layers.OutputLayer;

public class DL4JClassifierTest {
	
	public Instances loadIris() throws Exception {
		DataSource ds = new DataSource("datasets/iris.arff");
		Instances data = ds.getDataSet();
		data.setClassIndex( data.numAttributes() - 1 );
		return data;
	}
	
	public Instances loadDiabetes() throws Exception {
		DataSource ds = new DataSource("datasets/diabetes_numeric.arff");
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
	public void irisTest() throws Exception {
		int numIters = 10;
		Instances data = loadIris();
		Dl4jMlpClassifier cls = getMlp();
		cls.setTrainBatchSize(50);
		cls.setNumIterations(numIters);
		String tmpFile = System.getProperty("java.io.tmpdir") + File.separator + "irisTest.txt";
		System.err.println("irisTest() tmp file: " + tmpFile);
		cls.setDebugFile(tmpFile);
		cls.buildClassifier(data);
		cls.distributionsForInstances(data);
		List<String> lines = Files.readAllLines(new File(tmpFile).toPath());
		assertEquals(lines.size(), numIters+1);
	}
	
	@Ignore
	public void testCSVRecordReader() throws Exception {
		CSVRecordReader reader = new CSVRecordReader();
		reader.initialize(new FileSplit(new File("datasets/diabetes.csv")));
		DataSetIterator iter = new RecordReaderDataSetIterator(reader,null,100,2,1,true);
        DataSet next = iter.next();
        System.out.println(next);
	}
	
	@Test
	public void diabetesTest() throws Exception {
		int numIters = 10;
		Instances data = loadDiabetes();
		Dl4jMlpClassifier cls = new Dl4jMlpClassifier();
		weka.dl4j.layers.DenseLayer hiddenLayer = new weka.dl4j.layers.DenseLayer();
		hiddenLayer.setNumUnits(3);
		hiddenLayer.setActivation(Activation.RELU);
		weka.dl4j.layers.OutputLayer outputLayer = new weka.dl4j.layers.OutputLayer();
		outputLayer.setActivation(Activation.IDENTITY);
		outputLayer.setLossFunction(LossFunction.MSE);
		cls.setLayers(new weka.dl4j.layers.Layer[] { hiddenLayer, outputLayer });
		cls.setTrainBatchSize(50);
		cls.setNumIterations(numIters);
		String tmpFile = System.getProperty("java.io.tmpdir") + File.separator + "diabetesTest.txt";
		System.err.println("diabetesTest() tmp file: " + tmpFile);
		cls.setDebugFile(tmpFile);
		cls.buildClassifier(data);
		double[][] dist = cls.distributionsForInstances(data);
		//for(int x = 0; x < dist.length; x++)
		//	System.out.println(Arrays.toString(dist[x]));
		List<String> lines = Files.readAllLines(new File(tmpFile).toPath());
		assertEquals(lines.size(), numIters+1);
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
