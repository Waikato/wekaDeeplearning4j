package weka.classifiers.functions;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import weka.classifiers.RandomizableClassifier;
import weka.classifiers.functions.dl4j.Utils;
import weka.classifiers.rules.ZeroR;
import weka.core.BatchPredictor;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.dl4j.Constants;
import weka.dl4j.FileIterationListener;
import weka.dl4j.iterators.AbstractDataSetIterator;
import weka.dl4j.iterators.DefaultDataSetIterator;
//import weka.dl4j.iterators.ImageDataSetIterator;
import weka.dl4j.layers.Conv2DLayer;
import weka.dl4j.layers.DenseLayer;
import weka.dl4j.layers.Layer;
import weka.dl4j.layers.Pool2DLayer;
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
public class Dl4jMlpClassifier extends RandomizableClassifier implements BatchPredictor {
	
	private static final Logger log = LoggerFactory.getLogger(Dl4jMlpClassifier.class);

	private ReplaceMissingValues m_replaceMissing = null;
	private Filter m_normalize = null;
	private boolean m_standardizeInsteadOfNormalize = true;
	private NominalToBinary m_nominalToBinary = null;
	private ZeroR m_zeroR = new ZeroR();

	private MultiLayerNetwork m_model = null;

	private static final long serialVersionUID = -6363244115597574265L;
	
	public String globalInfo() {
		return "Create MLPs with DL4J";
	}
	
	protected String m_debugFile = "";

	public String getDebugFile() {
		return m_debugFile;
	}

	public void setDebugFile(String debugFile) {
		m_debugFile = debugFile;
	}
	
	protected boolean m_vis = false;
	
	public boolean getVisualisation() {
		return m_vis;
	}
	
	public void setVisualisation(boolean b) {
		m_vis = b;
	}

	private Layer[] m_layers = new Layer[] {};

	public void setLayers(Layer[] layers) {
		m_layers = layers;
	}

	@OptionMetadata(description = "Layers", displayName = "layers", displayOrder = 1)
	public Layer[] getLayers() {
		return m_layers;
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

	@OptionMetadata(description = "Optimisation algorithm", displayName = "optimizationAlgorithm", displayOrder = 1)
	public OptimizationAlgorithm getOptimizationAlgorithm() {
		return m_optimAlgorithm;
	}

	public void setOptimizationAlgorithm(OptimizationAlgorithm optimAlgorithm) {
		m_optimAlgorithm = optimAlgorithm;
	}

	private double m_learningRate = 0.01;

	@OptionMetadata(description = "Learning rate", displayName = "learningRate", displayOrder = 1)
	public double getLearningRate() {
		return m_learningRate;
	}

	public void setLearningRate(double learningRate) {
		m_learningRate = learningRate;
	}

	private double m_momentum = 0.9;

	@OptionMetadata(description = "Momentum", displayName = "momentum", displayOrder = 1)
	public double getMomentum() {
		return m_momentum;
	}

	public void setMomentum(double momentum) {
		m_momentum = momentum;
	}

	public Updater m_updater = Updater.NESTEROVS;

	@OptionMetadata(description = "Gradient descent update algorithm to use", displayName = "updater", displayOrder = 1)
	public Updater getUpdater() {
		return m_updater;
	}

	public void setUpdater(Updater updater) {
		m_updater = updater;
	}
	
	private AbstractDataSetIterator m_iterator = new DefaultDataSetIterator();
	
	public AbstractDataSetIterator getDataSetIterator() {
		return m_iterator;
	}
	
	public void setDataSetIterator(AbstractDataSetIterator iterator) {
		m_iterator = iterator;
	}
	
	private int m_numEpochs = 10;
	
	public void setNumEpochs(int numEpochs) {
		m_numEpochs = numEpochs;
	}
	
	public int getNumEpochs() {
		return m_numEpochs;
	}

	public void validate() throws Exception {
		if (m_layers.length == 0) {
			throw new Exception("No layers have been added!");
		}
		if( ! (m_layers[ m_layers.length-1 ] instanceof weka.dl4j.layers.OutputLayer) ) {
			throw new Exception("Last layer in network must be an output layer!");
		}
	}
	
	/**
	 * Get the current number of units (where "units" are
	 * feature maps for conv nets and the # of hidden units
	 * for dense layers) for a particular layer.
	 * @param layer
	 * @return
	 */
	public int getNumUnits(Layer layer) {
		if(layer instanceof DenseLayer) {
			DenseLayer tmp = (DenseLayer) layer;
			return tmp.getNumUnits();
		}
		return -1;
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
			// we want to also normalize the class
			m_normalize.setOptions(new String[] { "-unset-class-temporarily" } );
		} else {
			m_normalize = new Normalize();
		}
		m_normalize.setInputFormat(data);
		data = Filter.useFilter(data, m_normalize);
		m_nominalToBinary = new NominalToBinary();
		m_nominalToBinary.setInputFormat(data);
		data = Filter.useFilter(data, m_nominalToBinary);
		
		//if(getDebug()) {
		//	System.out.println(data);
		//}
		
		//data.randomize(new Random(getSeed()));
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
		int numInputAttributes = getDataSetIterator().getNumAttributes(data);
		

		
		for (int x = 0; x < m_layers.length; x++) {
			// output layer
			if ( x == m_layers.length-1 ) {
				// if this is the only layer (i.e. a perceptron)
				if( x == 0) {
					if( !(m_layers[x] instanceof Conv2DLayer )) m_layers[x].setNumIncoming(numInputAttributes);
					m_layers[x].setNumOutgoing(data.numClasses());
				} else { // otherwise
					//weka.dl4j.layers.DenseLayer prevLayer = (weka.dl4j.layers.DenseLayer) m_layers[x - 1];
					//m_layers[x].setNumIncoming(prevLayer.getNumUnits());
					if( !(m_layers[x] instanceof Conv2DLayer )) m_layers[x].setNumIncoming( getNumUnits(m_layers[x-1]) );
					m_layers[x].setNumOutgoing(data.numClasses());
					//if(getDebug()) System.err.format("layer %d has prev incoming: %d\n", x, nIn);
				}
				ip = ip.layer(x, m_layers[x].getLayer() );
			// if this is the first layer
			} else if (x == 0) {
				if( !(m_layers[x] instanceof Conv2DLayer )) m_layers[x].setNumIncoming(numInputAttributes);
				ip = ip.layer(x, m_layers[x].getLayer() );
			} else {
				if( !(m_layers[x] instanceof Conv2DLayer )) m_layers[x].setNumIncoming( getNumUnits(m_layers[x-1]) );
				ip = ip.layer(x, m_layers[x].getLayer() );
			}
		}
		ip = ip.pretrain(false).backprop(true);
		
		// if a conv network
//		if( getDataSetIterator() instanceof ImageDataSetIterator ) {
//			ImageDataSetIterator tmp = (ImageDataSetIterator) getDataSetIterator();
//			new ConvolutionLayerSetup(ip, tmp.getHeight(), tmp.getWidth(), tmp.getNumChannels());
//		}
		
		MultiLayerConfiguration conf = ip.build();
		
		if( getDebug() ) {
			System.err.println( conf.toJson() );
		}
		
		// build the network
		m_model = new MultiLayerNetwork(conf);
		m_model.init();
		
		ArrayList<IterationListener> listeners = new ArrayList<IterationListener>();
		listeners.add( new ScoreIterationListener( data.numInstances() / getDataSetIterator().getTrainBatchSize() ) );
		
		//System.out.println(conf);
		int numMiniBatches = (int) Math.ceil( ((double)data.numInstances()) / ((double)getDataSetIterator().getTrainBatchSize()) );
		// if the debug file doesn't point to a directory, set up the listener
		if( !getDebugFile().equals("") ) {
			listeners.add( new FileIterationListener(getDebugFile(), numMiniBatches) );
			//m_model.setListeners(new FileIterationListener(getDebugFile(), numMiniBatches));
		}
		
		if( getVisualisation() ) {
			listeners.add( new HistogramIterationListener(1) );
		}
		
		m_model.setListeners(listeners);
		
		// if debug mode is set, print the shape of the outputs
		// of the network
		if( getDebug() ) {
			DataSetIterator tmpIter = getDataSetIterator().getTrainIterator(data, getSeed());
	        while(tmpIter.hasNext()) {
	        	DataSet d = tmpIter.next();
    	        m_model.initialize(d);
    	        List<INDArray> activations = m_model.feedForward();
    	        for(int i = 0; i < activations.size(); i++) {
    	        	log.info("*** Output shape of layer {} is {} ***", (i+1), Arrays.toString(activations.get(i).shape()) );
    	        }
    	        System.out.format("number of params: %d\n", m_model.numParams() );
	        	break;
	        }
		}
		
		DataSetIterator iter = getDataSetIterator().getTrainIterator(data, getSeed());
		for(int i = 0; i < getNumEpochs(); i++) {
			m_model.fit(iter);
			log.info("*** Completed epoch {} ***", i+1);		
			iter.reset();
		}
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
		
		// TODO: implement distributionforinstances instead
		Instances insts = new Instances( inst.dataset() );
		insts.add(inst);
		DataSetIterator iter = getDataSetIterator().getTestIterator( insts, getSeed(), Integer.parseInt(getBatchSize()) );
		//while(iter.hasNext()) {
		INDArray testMatrix = iter.next().getFeatureMatrix();	
		//}
		
		INDArray predicted = m_model.output(testMatrix);
		
		//INDArray predicted = m_model.output(Utils.instanceToINDArray(inst));
		predicted = predicted.getRow(0);
		double[] preds = new double[inst.numClasses()];
		for (int i = 0; i < preds.length; i++) {
			preds[i] = predicted.getDouble(i);
		}
		// only normalise if we're dealing with classification
		if( preds.length > 1) {
			weka.core.Utils.normalize(preds);
		}
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
		result.add("-" + Constants.DATASET_ITERATOR);
		result.add("" + getSpec(getDataSetIterator()));
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
		// debug file
		if( ! new File(getDebugFile()).isDirectory() ) {
			result.add("-" + Constants.DEBUG_FILE);
			result.add( getDebugFile() );
		}
		// vis
		if( getVisualisation() ) {
			result.add("-" + Constants.VIS);
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
		
		String tmp = weka.core.Utils.getOption(Constants.DATASET_ITERATOR, options);
		if(!tmp.equals("")) setDataSetIterator( 
				(AbstractDataSetIterator) specToObject(tmp, weka.dl4j.iterators.AbstractDataSetIterator.class));

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
		// debug file
		tmp = weka.core.Utils.getOption(Constants.DEBUG_FILE, options);
		if(!tmp.equals("")) setDebugFile(tmp);
		// vis
		tmp = weka.core.Utils.getOption(Constants.VIS, options);
		if(!tmp.equals("")) setVisualisation(true);
	}
	
	@Override
	public String toString() {
		if(m_model != null) {
			return m_model.conf().toYaml();
		}
		return null;
	}
	

	public Enumeration<Option> listOptions() {
		Vector<Option> v = new Vector<Option>();
		// layers
		v.add(new Option(
			"\tList of layers that define the MLP/CNN",
			"-layers",
			1, 
			"-layers <layer arguments>"
		));
		// dataset iterator
		v.add(new Option(
				"\tDataset iterator to use",
				"-iterator",
				1, 
				"-iterator <iterator arguments>"
		));
		// optim alg
		String optNames = "(";
		for(int i = 0; i < OptimizationAlgorithm.values().length; i++) {
			optNames += ( OptimizationAlgorithm.values()[i].toString() + " | ");
		}
		optNames += ")";
		v.add(new Option(
				"\tOptimisation algorithm to use",
				"-optim",
				1, 
				"-optim " + optNames
		));
		// learning rate
		v.add(new Option(
				"\tLearning rate",
				"-lr",
				1, 
				"-lr <float>"
		));		
		// momentum
		v.add(new Option(
				"\tMomentum (note: this depends on the updater, e.g. this option has no effect if using plain SGD optimiser)",
				"-momentum",
				1, 
				"-momentum <float>"
		));
		// updater
		optNames = "(";
		for(int i = 0; i < Updater.values().length; i++) {
			optNames += ( Updater.values()[i].toString() + " | ");
		}
		optNames += ")";		
		v.add(new Option(
				"\tType of updater to use",
				"-updater",
				1, 
				"-updater " + optNames
			));
		// debug
		v.add(new Option(
			"\tDebug file to print training statistics to",
			"-debug",
			1, 
			"-debug <filename>"
		));
		
		
		return v.elements();

	}
	
	public static void main(String[] argv) {
		runClassifier(new Dl4jMlpClassifier(), argv);
	}

}
