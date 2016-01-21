package weka.classifiers.functions;

import java.io.File;
import java.util.Random;
import java.util.Vector;

import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import weka.classifiers.RandomizableClassifier;
import weka.classifiers.functions.dl4j.Utils;
import weka.classifiers.rules.ZeroR;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.dl4j.Constants;
import weka.dl4j.FileIterationListener;
import weka.dl4j.ShufflingDataSetIterator;
import weka.dl4j.layers.Layer;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;

/**
 * Chris Beckham's DL4J classifier.
 * 
 * @author cjb60
 */
public class Dl4jMlpClassifier extends RandomizableClassifier {

	private ReplaceMissingValues m_replaceMissing = null;
	private Filter m_normalize = null;
	private boolean m_standardizeInsteadOfNormalize = true;
	private NominalToBinary m_nominalToBinary = null;
	private ZeroR m_zeroR = new ZeroR();

	private MultiLayerNetwork m_model = null;

	private static final long serialVersionUID = -6363244115597574265L;
	
	protected String m_debugFile = System.getProperty("user.dir");
	
	public String getDebugFile() {
		return m_debugFile;
	}

	public void setDebugFile(String debugFile) {
		m_debugFile = debugFile;
	}

	private Layer[] m_layers = new Layer[] {};

	public void setLayers(Layer[] layers) {
		m_layers = layers;
	}

	@OptionMetadata(description = "Layers", displayName = "layers", displayOrder = 1)
	public Layer[] getLayers() {
		return m_layers;
	}
	
	private int m_trainBatchSize = 1;
	
	public void setTrainBatchSize(int trainBatchSize) {
		m_trainBatchSize = trainBatchSize;
	}
	
	public int getTrainBatchSize() {
		return m_trainBatchSize;
	}

	private int m_numIterations = 100;

	public void setNumIterations(int numIterations) {
		m_numIterations = numIterations;
	}

	public int getNumIterations() {
		return m_numIterations;
	}

	/*
	private GradientNormalization m_gradientNorm = GradientNormalization.None;

	public GradientNormalization getGradientNorm() {
		return m_gradientNorm;
	}

	public void setGradientNorm(GradientNormalization gradientNorm) {
		m_gradientNorm = gradientNorm;
	}
	*/

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

	public Updater m_updater = Updater.NESTEROVS;

	public Updater getUpdater() {
		return m_updater;
	}

	public void setUpdater(Updater updater) {
		m_updater = updater;
	}

	public void validate() throws Exception {
		if (m_layers.length == 0) {
			throw new Exception("No layers have been added!");
		}
		if( ! (m_layers[ m_layers.length-1 ] instanceof weka.dl4j.layers.OutputLayer) ) {
			throw new Exception("Last layer in network must be an output layer!");
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
				.seed(getSeed())
				.iterations(1)
				.learningRate(getLearningRate())
				.momentum(getMomentum())
				//.gradientNormalization(getGradientNorm())
				.optimizationAlgo(getOptimizationAlgorithm())
				.updater(getUpdater())
				//.batchSize( getTrainBatchSize() )
				.list(m_layers.length);
		for (int x = 0; x < m_layers.length; x++) {
			if (x == 0) {
				// input layer
				m_layers[x].setNumIncoming(data.numAttributes()-1);
				ip = ip.layer(x, m_layers[x].getLayer() );
			} else if ( x == m_layers.length-1 ) {
				// output layer
				weka.dl4j.layers.DenseLayer prevLayer = (weka.dl4j.layers.DenseLayer) m_layers[x - 1];
				m_layers[x].setNumIncoming(prevLayer.getNumUnits());
				m_layers[x].setNumOutgoing(data.numClasses());
				ip = ip.layer(x, m_layers[x].getLayer() );
			} else {
				// intermediate layer
				weka.dl4j.layers.DenseLayer prevLayer = (weka.dl4j.layers.DenseLayer) m_layers[x - 1];
				m_layers[x].setNumIncoming(prevLayer.getNumUnits());
				ip = ip.layer(x, m_layers[x].getLayer());
			}
		}
		ip = ip.pretrain(false).backprop(true);
		MultiLayerConfiguration conf = ip.build();
		// build the network
		m_model = new MultiLayerNetwork(conf);
		m_model.init();
		int numMiniBatches = (int) Math.ceil( ((double)dataset.numExamples()) / ((double)getTrainBatchSize()) );
		// if the debug file doesn't point to a directory, set up the listener
		if( ! new File(getDebugFile()).isDirectory() ) {
			m_model.setListeners(new FileIterationListener(getDebugFile(), numMiniBatches));
		}
		// train
		MultipleEpochsIterator iter = new MultipleEpochsIterator(
				getNumIterations()-1, 
				new ShufflingDataSetIterator(dataset, getTrainBatchSize(), getSeed()));
		m_model.fit(iter);
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
						+ weka.core.Utils.joinOptions(((OptionHandler) obj)
								.getOptions());
			}
		}
		return result;
	}

	public static Object specToObject(String str, Class<?> classType)
			throws Exception {
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
			result.add(getSpec(getLayers()[i]));
		}
		// num iterations
		result.add("-" + Constants.NUM_ITERATIONS);
		result.add("" + getNumIterations());
		// gradient norm
		//result.add("-" + Constants.GRADIENT_NORM);
		//result.add("" + getGradientNorm().name());
		// optimization algorithm
		result.add("-" + Constants.OPTIMIZATION_ALGORITHM);
		result.add("" + getOptimizationAlgorithm().name());
		// learning rate
		result.add("-" + Constants.LEARNING_RATE);
		result.add("" + getLearningRate());
		// momentum
		result.add("-" + Constants.MOMENTUM);
		result.add("" + getMomentum());
		// loss function
		//result.add("-" + Constants.LOSS_FUNCTION);
		//result.add("" + getLossFunction().name());
		// updater
		result.add("-" + Constants.UPDATER);
		result.add("" + getUpdater().name());
		// train batch size
		result.add("-" + Constants.TRAIN_BATCH_SIZE);
		result.add("" + getTrainBatchSize());
		// debug file
		if( ! new File(getDebugFile()).isDirectory() ) {
			result.add("-" + Constants.DEBUG_FILE);
			result.add( getDebugFile() );
		}

		return result.toArray(new String[result.size()]);
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		super.setOptions(options);
		// layers
		Vector<weka.dl4j.layers.Layer> layers = new Vector<weka.dl4j.layers.Layer>();
		String tmpStr = null;
		while ((tmpStr = weka.core.Utils.getOption(Constants.LAYER, options))
				.length() != 0) {
			layers.add((weka.dl4j.layers.Layer) specToObject(tmpStr, weka.dl4j.layers.Layer.class));
		}
		if (layers.size() == 0) {
			layers.add(new weka.dl4j.layers.DenseLayer());
		}
		setLayers(layers.toArray(new weka.dl4j.layers.Layer[layers.size()]));
		// num iterations
		String tmp = weka.core.Utils.getOption(Constants.NUM_ITERATIONS, options);
		if(!tmp.equals("")) setNumIterations( Integer.parseInt(tmp) );
		// gradient norm
		//
		//
		// optimization algorithm
		tmp = weka.core.Utils.getOption(Constants.OPTIMIZATION_ALGORITHM, options);
		if(!tmp.equals("")) setOptimizationAlgorithm( OptimizationAlgorithm.valueOf(tmp) );
		// learning rate
		tmp = weka.core.Utils.getOption(Constants.LEARNING_RATE, options);
		if(!tmp.equals("")) setLearningRate( Double.parseDouble(tmp) );
		// momentum
		tmp = weka.core.Utils.getOption(Constants.MOMENTUM, options);
		if(!tmp.equals("")) setMomentum( Double.parseDouble(tmp) );
		// loss function
		//tmp = weka.core.Utils.getOption(Constants.LOSS_FUNCTION, options);
		//if(!tmp.equals("")) setLossFunction( LossFunction.valueOf(tmp) );
		// updater
		tmp = weka.core.Utils.getOption(Constants.UPDATER, options);
		if(!tmp.equals("")) setUpdater( Updater.valueOf(tmp) );
		// train batch size
		tmp = weka.core.Utils.getOption(Constants.TRAIN_BATCH_SIZE, options);
		if(!tmp.equals("")) setTrainBatchSize( Integer.parseInt(tmp) );
		// debug file
		tmp = weka.core.Utils.getOption(Constants.DEBUG_FILE, options);
		if(!tmp.equals("")) setDebugFile(tmp);
	}
	
	@Override
	public String toString() {
		if(m_model != null) {
			return m_model.conf().toYaml();
		}
		return null;
	}

}
