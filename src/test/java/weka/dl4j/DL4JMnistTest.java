package weka.dl4j;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class DL4JMnistTest {
	
	public static void main(String[] args) throws Exception {
		
		DataSetIterator iter = new MnistDataSetIterator(1000,1000);
		DataSet data = iter.next();
		System.out.println(data.numExamples());
		System.out.println(data.numInputs());
		System.out.println(data.numOutcomes());
		
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
        .seed(0)
        .iterations(10)
        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .list(3)
        .layer(0, new ConvolutionLayer.Builder(10, 10)
                .stride(2,2)
                .nIn(1)
                .nOut(6)
                .weightInit(WeightInit.XAVIER)
                .activation("relu")
                .build())
        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2,2})
                .build())
        .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10)
                .weightInit(WeightInit.XAVIER)
                .activation("softmax")
                .build())
        .backprop(true).pretrain(false);

		new ConvolutionLayerSetup(builder, 28, 28, 1);
		
		MultiLayerConfiguration conf = builder.build();
		
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		
		model.fit(data);
		
		
		
		
		
		
	}

}
