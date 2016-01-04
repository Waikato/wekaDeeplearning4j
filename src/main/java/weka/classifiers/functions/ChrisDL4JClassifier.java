package weka.classifiers.functions;

import java.util.Random;

import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.dataset.DataSet;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.RandomizableClassifier;
import weka.classifiers.functions.dl4j.Utils;
import weka.classifiers.rules.ZeroR;
import weka.core.Instances;
import weka.core.OptionMetadata;
import weka.dl4j.layers.Layer;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;


/**
 * Chris Beckham's DL4J classifier.
 * @author cjb60
 */
public class ChrisDL4JClassifier extends RandomizableClassifier {
	
	private ReplaceMissingValues m_replaceMissing = null;
	private Filter m_normalize = null;
	private boolean m_standardizeInsteadOfNormalize = true;
	private NominalToBinary m_nominalToBinary = null;
	private ZeroR m_zeroR = new ZeroR();

	private static final long serialVersionUID = -6363244115597574265L;
	
	private Layer[] m_layers = new Layer[] { };
	
	public void setLayers(Layer[] layers) {
		m_layers = layers;
	}
	
	@OptionMetadata(description = "Layers", displayName = "layers", displayOrder=1)
	public Layer[] getLayers() {
		return m_layers;
	}
	
	private int m_numIterations = 100;
	
	public void setNumIterations(int numIterations) {
		m_numIterations = numIterations;
	}
	
	public int getNumIterations() {
		return m_numIterations;
	}
	
	private GradientNormalization m_gradientNorm = GradientNormalization.None;
	
	public GradientNormalization getGradientNorm() {
		return m_gradientNorm;
	}
	
	public void setGradientNorm(GradientNormalization gradientNorm) {
		m_gradientNorm = gradientNorm;
	}
	
	private double m_learningRate = 0.01;
	
	public double getLearningRate() {
		return m_learningRate;
	}
	
	public void setLearningRate(double learningRate) {
		m_learningRate = learningRate;
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		m_zeroR = null;
		// remove instances with missing class
		data = new Instances(data);
		data.deleteWithMissingClass();
		if (data.numInstances() == 0 || data.numAttributes() < 2) {
			m_zeroR.buildClassifier(data);
			return;
		}
		// can classifier handle the data?
		getCapabilities().testWithFail(data);
		m_replaceMissing = new ReplaceMissingValues();
		m_replaceMissing.setInputFormat(data);
		data = Filter.useFilter(data, m_replaceMissing);
		if (m_standardizeInsteadOfNormalize) {
			m_normalize = new Standardize();
		} else {
			m_normalize = new Normalize();
		}
		m_normalize.setInputFormat(data);
		data = Filter.useFilter(data, m_normalize);
		m_nominalToBinary = new NominalToBinary();
		m_nominalToBinary.setInputFormat(data);
		data = Filter.useFilter(data, m_nominalToBinary);
		data.randomize(new Random(123));
		// convert the dataset
		DataSet dataset = Utils.instancesToDataSet(data);
		// construct the mlp configuration
		ListBuilder ip = new NeuralNetConfiguration.Builder()
			.seed( getSeed() )
			.iterations( getNumIterations() )
			.learningRate( getLearningRate() )
			.gradientNormalization( getGradientNorm() )
			.list(1);
		ip = ip.layer(0, new DenseLayer.Builder().nIn( data.numAttributes()-1 ).nOut(3)
            .activation("tanh")
            .weightInit(WeightInit.XAVIER)
            .build());
		ip = ip.backprop(true);
		MultiLayerConfiguration conf = ip.build();
		
	}
	
	
	
	/*
	 *         MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	                .seed(seed)
	                .iterations(iterations)

	                .learningRate(1e-3)
	                .l1(0.3).regularization(true).l2(1e-3)
	                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
	                .list(3)
	                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(3)
	                        .activation("tanh")
	                        .weightInit(WeightInit.XAVIER)
	                        .build())
	                .layer(1, new DenseLayer.Builder().nIn(3).nOut(2)
	                        .activation("tanh")
	                        .weightInit(WeightInit.XAVIER)
	                        .build())
	                .layer(2, new OutputLayer.Builder(LossFunction.MCXENT)
	                        .weightInit(WeightInit.XAVIER)
	                        .activation("softmax")
	                		.nIn(2).nOut(outputNum).build())
	                .backprop(true).pretrain(false)
	                .build();
	 * 
	 */
	
	

}
