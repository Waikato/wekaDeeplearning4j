package weka.classifiers.functions;

import java.util.Arrays;
import java.util.Collections;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import weka.classifiers.functions.dl4j.Utils;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class DL4JIrisSimple {
	
	public static void main(String[] args) throws Exception {
		
		DataSource ds = new DataSource("/Users/cjb60/github/weka-pyscript/datasets/iris.arff");
		Instances iris = ds.getDataSet();
		iris.setClassIndex( iris.numAttributes() - 1 );
		
		DataSet dataset = Utils.instancesToDataSet(iris);
		
		int numIters = 100;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	        .seed(1)
	        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	        .iterations(numIters)
	        .learningRate(0.01)
	        .momentum(0.9)
	        .list(2)
	        .layer(0, new DenseLayer.Builder()
	        	.nIn(iris.numAttributes()-1)
	        	.nOut(10)
	        	.activation("relu")
	        	.weightInit(WeightInit.XAVIER)
	        	.build()
	        )
	        .layer(1, new OutputLayer.Builder(LossFunction.MCXENT)
	        	.nIn(10)
	        	.nOut(iris.numClasses())
	        	.activation("softmax")
	        	.weightInit(WeightInit.XAVIER)
	        	.build()
	        )
	        .build();
		
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(1)));
        
        model.fit(dataset);
        
        

        
	}

}
