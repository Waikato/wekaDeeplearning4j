package weka.classifiers.functions;

import java.util.Random;
import java.util.Vector;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import weka.classifiers.RandomizableClassifier;
import weka.classifiers.functions.dl4j.Utils;
import weka.classifiers.rules.ZeroR;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.dl4j.layers.Constants;
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
	
	private MultiLayerNetwork m_model = null;

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
	
	private OptimizationAlgorithm m_optimAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
	
	public OptimizationAlgorithm getOptimizationAlgorithm() {
		return m_optimAlgorithm;
	}
	
	public void setOptimizationAlgorithm(OptimizationAlgorithm optimAlgorithm) {
		m_optimAlgorithm = optimAlgorithm;
	}
	
	private double m_learningRate = 0.01;
	
	public double getLearningRate() {
		return m_learningRate;
	}
	
	public void setLearningRate(double learningRate) {
		m_learningRate = learningRate;
	}
	
	private double m_momentum = 0.9;
	
	public double getMomentum() {
		return m_momentum;
	}
	
	public void setMomentum(double momentum) {
		m_momentum = momentum;
	}
	
	private LossFunction m_lossFunction = LossFunction.MCXENT;
	
	public LossFunction getLossFunction() {
		return m_lossFunction;
	}
	
	public void setLossFunction(LossFunction lossFunction) {
		m_lossFunction = lossFunction;
	}
	
	public Updater m_updater = Updater.NESTEROVS;
	
	public Updater getUpdater() {
		return m_updater;
	}
	
	public void setUpdater(Updater updater) {
		m_updater = updater;
	}

	public void validate() throws Exception {
		if(m_layers.length == 0) {
			throw new Exception("No layers have been added!");
		}
	}
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		// validate
		validate();
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
			.optimizationAlgo( getOptimizationAlgorithm() )
			.updater( getUpdater() )
			.list( m_layers.length+1 );
		
		for(int x = 0; x < m_layers.length; x++) {
			if(x == 0) {
				ip = ip.layer(x, m_layers[x].getLayer(x, data.numAttributes()-1));
			} else {
				weka.dl4j.layers.DenseLayer prevLayer = (weka.dl4j.layers.DenseLayer) m_layers[x-1];
				ip = ip.layer( x, m_layers[x].getLayer(x, prevLayer.getNumUnits()) );
			}
		}

		// TODO: make output layer selectable
		
		weka.dl4j.layers.DenseLayer prevLayer = (weka.dl4j.layers.DenseLayer) m_layers[ m_layers.length - 1 ];
		ip = ip.layer(m_layers.length, new OutputLayer.Builder( getLossFunction() )
        	.weightInit(WeightInit.XAVIER)
        	.activation("softmax")
        	.nIn( prevLayer.getNumUnits() ).nOut( data.numClasses() )
        	.build());
		ip = ip.backprop(true);
		MultiLayerConfiguration conf = ip.build();
		// build the network
		m_model = new MultiLayerNetwork(conf);
		m_model.init();
        // train
		m_model.fit(dataset);
	}
	
	public double[] distributionForInstance(Instance inst) throws Exception {
		if (m_zeroR != null) {
			return m_zeroR.distributionForInstance(inst);
		}
		m_replaceMissing.input(inst);
		inst = m_replaceMissing.output();
		m_normalize.input(inst);
		inst = m_normalize.output();
		m_nominalToBinary.input(inst);
		inst = m_nominalToBinary.output();
		INDArray predicted = m_model.output(Utils.instanceToINDArray(inst));
		predicted = predicted.getRow(0);
		double[] preds = new double[inst.numClasses()];
		for (int i = 0; i < preds.length; i++) {
			preds[i] = predicted.getDouble(i);
		}
		weka.core.Utils.normalize(preds);
		return preds;
	}
	
	public static String getSpec(Object obj) {
		String result;
		if (obj == null) {
			result = "";
		} else {
			result = obj.getClass().getName();
			if (obj instanceof OptionHandler) {
				result += " "
		          + weka.core.Utils.joinOptions(((OptionHandler) obj).getOptions());
			}
		}		
		return result;
	}
	
	public static Object specToObject(String str, Class<?> classType) throws Exception {	
		String[] options = weka.core.Utils.splitOptions(str);
		String base = options[0];
		options[0] = "";
		return weka.core.Utils.forName(classType, base, options);
	}
	
	@Override
	public String[] getOptions() {
	    Vector<String> result = new Vector<String>();
	    String[] options = super.getOptions();
	    for (int i = 0; i < options.length; i++) {
	      result.add(options[i]);
	    }
	    // layers
	    for (int i = 0; i < getLayers().length; i++) {
	      result.add("-" + Constants.LAYER);
	      result.add( getSpec(getLayers()[i]) );
	    }
	    // num iterations
	    result.add("-" + Constants.NUM_ITERATIONS);
	    result.add( "" + getNumIterations() );
	    // gradient norm
	    result.add("-" + Constants.GRADIENT_NORM);
	    result.add( "" + getGradientNorm().name() );
	    // optimization algorithm
	    result.add("-" + Constants.OPTIMIZATION_ALGORITHM);
	    result.add( "" + getOptimizationAlgorithm().name());
	    // learning rate
	    result.add("-" + Constants.LEARNING_RATE);
	    result.add( "" + getLearningRate() );
	    // momentum
	    result.add("-" + Constants.MOMENTUM);
	    result.add( "" + getMomentum() );   
	    // loss function
	    result.add("-" + Constants.LOSS_FUNCTION);
	    result.add( "" + getLossFunction().name() );
	    // updater
	    result.add("-" + Constants.UPDATER);
	    result.add( "" + getUpdater().name() );	    
	    
	    return result.toArray(new String[result.size()]);
	}
	
	@Override
	public void setOptions(String[] options) throws Exception {
		super.setOptions(options);
		// layers
		Vector<Layer> layers = new Vector<Layer>();
		String tmpStr = null;
		while ((tmpStr = weka.core.Utils.getOption(Constants.LAYER, options)).length() != 0) {
			layers.add( (Layer) specToObject(tmpStr, Layer.class) );
		}
		if (layers.size() == 0) {
			layers.add(new weka.dl4j.layers.DenseLayer());
		}
		setLayers( layers.toArray(new Layer[layers.size()]) );
	}

}
