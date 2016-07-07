package weka.classifiers.functions;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;

import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.dl4j.Activation;
import weka.dl4j.iterators.AbstractDataSetIterator;
import weka.dl4j.iterators.DefaultDataSetIterator;
import weka.dl4j.iterators.ImageDataSetIterator;
import weka.dl4j.layers.Layer;

public class SpecificTest {
	
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
		cls.setDebug(true);
		cls.setLayers(new weka.dl4j.layers.Layer[] {
				new weka.dl4j.layers.DenseLayer(),
				new weka.dl4j.layers.DenseLayer(),
				new weka.dl4j.layers.OutputLayer() 
		});
		return cls;
	}
	
	public int findOne(INDArray arr) {
		for(int x = 0; x < arr.columns(); x++) {
			if( arr.getFloat(x) == 1.0 ) {
				return x;
			}
		}
		return -1;
	}
	
	@Test
	public void testMinimalMnistConvNet() throws Exception {
		Dl4jMlpClassifier cls = new Dl4jMlpClassifier();
		cls.setSeed(1);
		cls.setDebug(true); // want to see the network output shapes
		cls.setDebugFile("/tmp/debug.txt");
		DataSource ds = new DataSource("datasets/mnist.meta.arff");
		Instances data = ds.getDataSet();
		data.setClassIndex( data.numAttributes() - 1 );
		ImageDataSetIterator imgIter = new ImageDataSetIterator();
		imgIter.setImagesLocation(new File("mnist-data").getAbsolutePath());
		imgIter.setHeight(28);
		imgIter.setWidth(28);
		imgIter.setNumChannels(1);
		imgIter.setNumIterations(10);
		imgIter.setTrainBatchSize(128);
		cls.setDataSetIterator(imgIter);
		
		weka.dl4j.layers.Conv2DLayer convLayer = new weka.dl4j.layers.Conv2DLayer();
		convLayer.setNumFilters(16);
		convLayer.setFilterSizeX(3);
		convLayer.setFilterSizeY(3);
		convLayer.setStrideX(1);
		convLayer.setStrideY(1);
		convLayer.setActivation(Activation.RELU);
		convLayer.setWeightInit(WeightInit.XAVIER);
		
		weka.dl4j.layers.Pool2DLayer poolLayer = new weka.dl4j.layers.Pool2DLayer();
		poolLayer.setPoolSizeX(2);
		poolLayer.setPoolSizeY(2);
		poolLayer.setStrideX(2);
		poolLayer.setStrideY(2);
		
		weka.dl4j.layers.Conv2DLayer convLayer2 = new weka.dl4j.layers.Conv2DLayer();
		convLayer2.setNumFilters(32);
		convLayer2.setFilterSizeX(3);
		convLayer2.setFilterSizeY(3);
		convLayer2.setStrideX(1);
		convLayer2.setStrideY(1);
		convLayer2.setActivation(Activation.RELU);
		convLayer2.setWeightInit(WeightInit.XAVIER);
		
		weka.dl4j.layers.Pool2DLayer poolLayer2 = new weka.dl4j.layers.Pool2DLayer();
		poolLayer2.setPoolSizeX(2);
		poolLayer2.setPoolSizeY(2);
		poolLayer2.setStrideX(2);
		poolLayer2.setStrideY(2);
		
		weka.dl4j.layers.Conv2DLayer convLayer3 = new weka.dl4j.layers.Conv2DLayer();
		convLayer3.setNumFilters(48);
		convLayer3.setFilterSizeX(3);
		convLayer3.setFilterSizeY(3);
		convLayer3.setStrideX(1);
		convLayer3.setStrideY(1);
		convLayer3.setActivation(Activation.RELU);
		convLayer3.setWeightInit(WeightInit.XAVIER);
		
		weka.dl4j.layers.Pool2DLayer poolLayer3 = new weka.dl4j.layers.Pool2DLayer();
		poolLayer3.setPoolSizeX(2);
		poolLayer3.setPoolSizeY(2);
		poolLayer3.setStrideX(2);
		poolLayer3.setStrideY(2);
		
		weka.dl4j.layers.DenseLayer denseLayer = new weka.dl4j.layers.DenseLayer();
		denseLayer.setNumUnits(128);
		denseLayer.setActivation(Activation.RELU);
		denseLayer.setWeightInit(WeightInit.XAVIER);
		
		weka.dl4j.layers.OutputLayer outputLayer = new weka.dl4j.layers.OutputLayer();
		outputLayer.setActivation(Activation.SOFTMAX);
		outputLayer.setWeightInit(WeightInit.XAVIER);
		
		cls.setLearningRate(0.01);
		cls.setMomentum(0.9);
		cls.setUpdater(Updater.NESTEROVS);
		
		cls.setLayers( new Layer[] { convLayer, poolLayer, convLayer2, poolLayer2, convLayer3, poolLayer3, denseLayer, outputLayer } );		
		cls.buildClassifier(data);
		cls.distributionsForInstances(data);
		
		
/*		java -Xmx5g -cp $WEKA_HOME/weka.jar weka.Run \
	     .Dl4jMlpClassifier -S 1 \
	     -layer "weka.dl4j.layers.Conv2DLayer -num_filters 16 -filter_size_x 3 -filter_size_y 3 -stride_x 1 -stride_y 1 -activation relu -init XAVIER" \
	     -layer "weka.dl4j.layers.Pool2DLayer -pool_size_x 2 -pool_size_y 2 -stride_x 2 -stride_y 2 -pool_type max" \
	     -layer "weka.dl4j.layers.Conv2DLayer -num_filters 32 -filter_size_x 3 -filter_size_y 3 -stride_x 1 -stride_y 1 -activation relu -init XAVIER" \
	     -layer "weka.dl4j.layers.Pool2DLayer -pool_size_x 2 -pool_size_y 2 -stride_x 2 -stride_y 2 -pool_type max" \
	     -layer "weka.dl4j.layers.Conv2DLayer -num_filters 48 -filter_size_x 3 -filter_size_y 3 -stride_x 1 -stride_y 1 -activation relu -init XAVIER" \
	     -layer "weka.dl4j.layers.Pool2DLayer -pool_size_x 2 -pool_size_y 2 -stride_x 2 -stride_y 2 -pool_type max" \
	     -layer "weka.dl4j.layers.DenseLayer -units 128 -activation relu -init XAVIER" \
	     -layer "weka.dl4j.layers.OutputLayer -units 10 -activation softmax -init XAVIER -p 0.0 -l1 0.0 -l2 0.0 -loss MCXENT" \
	     -iterator "weka.dl4j.iterators.ImageDataSetIterator -bs 32 -iters 100 -width 28 -height 28 -channels 1 -location ../mnist-data" \
	     -optim STOCHASTIC_GRADIENT_DESCENT \
	     -lr 0.1 -momentum 0.9 -updater SGD -debug /tmp/debug.txt \
	     -output-debug-info \
	     -t ../datasets/mnist.meta.arff \
	     -batch-size 100 \
	     -no-cv */

		
	}

}
