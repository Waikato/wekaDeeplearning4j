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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import weka.classifiers.functions.dl4j.Utils;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.dl4j.Activation;
import weka.dl4j.iterators.ImageDataSetIterator;
import weka.dl4j.layers.Layer;
import weka.dl4j.layers.OutputLayer;

public class DL4JClassifierTest {
	
	@Test
	public void testFish() throws Exception {
		DataSource ds = new DataSource("datasets-numeric/fishcatch.arff");
		Instances data = ds.getDataSet();
		data.setClassIndex(data.numAttributes()-1);
		Dl4jMlpClassifier cls = new Dl4jMlpClassifier();
		OutputLayer out = new weka.dl4j.layers.OutputLayer();
		out.setLossFunction(LossFunction.MSE);
		cls.setLayers(new weka.dl4j.layers.Layer[] { out } );
		cls.setDebugFile("/tmp/debug.txt");
		cls.setLearningRate(0.01);
		cls.getDataSetIterator().setNumIterations(100);
		cls.getDataSetIterator().setTrainBatchSize(1000);
		cls.buildClassifier(data);
	}
	
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
	
	public Instances getMnistMeta() throws Exception {
		DataSource ds = new DataSource("datasets/mnist.meta.arff");
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
	 * Test a no-hidden-layer neural net (i.e. a perceptron)
	 * on the numeric diabetes dataset
	 * @throws Exception
	 */
	@Test
	public void testPerceptron() throws Exception {
		Dl4jMlpClassifier cls = new Dl4jMlpClassifier();
		cls.setLayers(new weka.dl4j.layers.Layer[] {
				new weka.dl4j.layers.OutputLayer() 
		});		
		Instances data = loadDiabetes();
		cls.buildClassifier(data);
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
		cls.getDataSetIterator().setTrainBatchSize(50);
		cls.getDataSetIterator().setNumIterations(numIters);
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
		cls.getDataSetIterator().setTrainBatchSize(50);
		cls.getDataSetIterator().setNumIterations(numIters);
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
	
	public int findOne(INDArray arr) {
		for(int x = 0; x < arr.columns(); x++) {
			if( arr.getFloat(x) == 1.0 ) {
				return x;
			}
		}
		return -1;
	}
	
	/**
	 * Test the image dataset iterator with a very
	 * minimal MNIST example (10 images, and 1 image
	 * per class), with 10 iterations and a mini-batch
	 * size of 3. We expect to get 100 images back,
	 * with each class containing 10 images.
	 * @throws Exception
	 */
	@Test
	public void testImageLoading() throws Exception {
		DataSource ds = new DataSource("datasets/mnist.meta.minimal.arff");
		Instances data = ds.getDataSet();
		data.setClassIndex(data.numAttributes()-1);
		ImageDataSetIterator imgIter = new ImageDataSetIterator();
		imgIter.setHeight(28);
		imgIter.setWidth(28);
		imgIter.setNumChannels(1);
		imgIter.setNumIterations(10);
		imgIter.setTrainBatchSize(3);
		imgIter.setNumIterations(10);
		imgIter.setImagesLocation(new File("datasets/mnist-minimal").getAbsolutePath());
		DataSetIterator iter = imgIter.getIterator(data, 0);
		int[] classCounts = new int[10];
		while(iter.hasNext()) {
			DataSet batch = iter.next();
			INDArray classes = batch.getLabels();
			for(int i = 0; i < classes.rows(); i++) {
				INDArray row = classes.getRow(i);
				classCounts[ findOne(row) ] += 1;
			}
		}
		for(int x = 0; x < classCounts.length; x++) {
			assertEquals(classCounts[x], 10);
		}
		
	}
	
	@Test
	public void testMinimalMnistDenseNet() throws Exception {
		Dl4jMlpClassifier cls = new Dl4jMlpClassifier();
		DataSource ds = new DataSource("datasets/mnist.meta.minimal.arff");
		Instances data = ds.getDataSet();
		data.setClassIndex( data.numAttributes() - 1 );
		ImageDataSetIterator imgIter = new ImageDataSetIterator();
		imgIter.setImagesLocation(new File("datasets/mnist-minimal").getAbsolutePath());
		imgIter.setHeight(28);
		imgIter.setWidth(28);
		imgIter.setNumChannels(1);
		imgIter.setNumIterations(10);
		imgIter.setTrainBatchSize(128);
		cls.setDataSetIterator(imgIter);
		weka.dl4j.layers.DenseLayer hiddenLayer = new weka.dl4j.layers.DenseLayer();
		//hiddenLayer.setNumIncoming(1*28*28);
		hiddenLayer.setNumUnits(10);
		hiddenLayer.setActivation(Activation.RELU);
		weka.dl4j.layers.OutputLayer outputLayer = new weka.dl4j.layers.OutputLayer();
		outputLayer.setActivation(Activation.SOFTMAX);
		cls.setLayers( new Layer[] { hiddenLayer, outputLayer } );		
		cls.buildClassifier(data);
	}
	
	@Test
	public void testMinimalMnistConvNet() throws Exception {
		Dl4jMlpClassifier cls = new Dl4jMlpClassifier();
		cls.setDebug(true); // want to see the network output shapes
		DataSource ds = new DataSource("datasets/mnist.meta.minimal.arff");
		Instances data = ds.getDataSet();
		data.setClassIndex( data.numAttributes() - 1 );
		ImageDataSetIterator imgIter = new ImageDataSetIterator();
		imgIter.setImagesLocation(new File("datasets/mnist-minimal").getAbsolutePath());
		imgIter.setHeight(28);
		imgIter.setWidth(28);
		imgIter.setNumChannels(1);
		imgIter.setNumIterations(10);
		imgIter.setTrainBatchSize(128);
		cls.setDataSetIterator(imgIter);
		weka.dl4j.layers.Conv2DLayer convLayer = new weka.dl4j.layers.Conv2DLayer();
		convLayer.setNumFilters(10);
		convLayer.setFilterSizeX(5);
		convLayer.setFilterSizeY(5);
		weka.dl4j.layers.Pool2DLayer poolLayer = new weka.dl4j.layers.Pool2DLayer();
		poolLayer.setPoolSizeX(2);
		poolLayer.setPoolSizeY(2);
		poolLayer.setStrideX(2);
		poolLayer.setStrideY(2);
		weka.dl4j.layers.OutputLayer outputLayer = new weka.dl4j.layers.OutputLayer();
		outputLayer.setActivation(Activation.SOFTMAX);
		cls.setLayers( new Layer[] { convLayer, poolLayer, outputLayer } );		
		cls.buildClassifier(data);
	}

}
