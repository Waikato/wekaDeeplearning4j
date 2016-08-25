package weka.classifiers.functions;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import weka.classifiers.RandomizableClassifier;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.timeseries.core.BaseModelSerializer;
import weka.classifiers.timeseries.core.StateDependentPredictor;
import weka.core.*;
import weka.dl4j.Constants;
import weka.dl4j.iterators.AbstractDataSetIterator;
import weka.dl4j.iterators.DefaultDataSetIterator;
import weka.dl4j.layers.LSTM;
import weka.dl4j.layers.Layer;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;

import java.io.*;
import java.util.*;

import static weka.classifiers.functions.dl4j.Utils.RNNinstancesToDataSet;

/**
 * Created by pedro on 18-07-2016. Time Series package must be in the classpath!
 */
public class Dl4jRNNForecaster extends RandomizableClassifier implements StateDependentPredictor, BaseModelSerializer {
    private static final long serialVersionUID = -7363244115597574265L;

    private ReplaceMissingValues m_replaceMissing = null;
    private Filter m_normalize = null;
    private boolean m_standardizeInsteadOfNormalize = false;
    private NominalToBinary m_nominalToBinary = null;
    private ZeroR m_zeroR = new ZeroR();

    private MultiLayerNetwork m_model = null;

    public String globalInfo() {
        return "Create RNNs with DL4J. This implementation uses Graves' LSTM structure to " +
                "deal with the vanishing/exploding gradient problem. These networks are indicated for time series " +
                "datasets and their output layer must be an RNNOutputLayer.";
    }

    // Serialization
    public void serializeModel(String path) throws IOException {
        File file = new File(path);
        ModelSerializer.writeModel(m_model, file, true);
    }

    // Serialize model state
    public void serializeState(String path) throws Exception {
        File sFile = new File(path);
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(sFile));
        List<Map<String, INDArray>> states = getPreviousState();
        oos.writeObject(states);
        oos.close();
    }

    // De-serialization
    public void loadSerializedModel(String path) throws IOException {
        File sFile = new File(path);
        m_model = ModelSerializer.restoreMultiLayerNetwork(sFile);
    }

    // De-serialize model state
    public void loadSerializedState(String path) throws Exception {
        File sFile = new File(path);
        ObjectInputStream is = new ObjectInputStream(new FileInputStream(sFile));
        Object states = (List<Map<String, INDArray>>) is.readObject();
        is.close();
        setPreviousState(states);
    }

    // Debug
    protected String m_debugFile = "";

    public String getDebugFile() {
        return m_debugFile;
    }

    public void setDebugFile(String debugFile) {
        m_debugFile = debugFile;
    }

    // Allow DL4J metrics visualization to help user in net configuration
    protected boolean m_vis = false;

    public boolean getVisualisation() {
        return m_vis;
    }

    public void setVisualisation(boolean b) {
        m_vis = b;
    }

    // Layers
    private Layer[] m_layers = new Layer[] {};

    public void setLayers(Layer[] layers) {
        m_layers = layers;
    }

    @OptionMetadata(description = "Layers", displayName = "layers", displayOrder = 1)
    public Layer[] getLayers() {
        return m_layers;
    }

    // Optimization algorithm
	private OptimizationAlgorithm m_optimAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;

    @OptionMetadata(description = "Optimization algorithm", displayName = "optimizationAlgorithm", displayOrder = 1)
    public OptimizationAlgorithm getOptimizationAlgorithm() {
        return m_optimAlgorithm;
    }

    public void setOptimizationAlgorithm(OptimizationAlgorithm optimAlgorithm) {
        m_optimAlgorithm = optimAlgorithm;
    }

    // Gradient normalization
    private GradientNormalization m_gradNorm = GradientNormalization.ClipL2PerLayer;

    @OptionMetadata(description = "Gradient Normalization", displayName = "gradientNormalization", displayOrder = 1)
    public GradientNormalization getGradientNormalization() {
        return m_gradNorm;
    }

    public void setGradientNormalization(GradientNormalization gradientNormalization) {
        m_gradNorm = gradientNormalization;
    }

    // Gradient normalization threshold
    private double m_gradNormThres = 0;

    @OptionMetadata(description = "Gradient Normalization Threshold", displayName = "gradientNormalizationThreshold", displayOrder = 1)
    public double getGradientNormalizationThres() {
        return m_gradNormThres;
    }

    public void setGradientNormalizationThres(double gradientNormalizationThres) {
        m_gradNormThres = gradientNormalizationThres;
    }

    // Number of epochs
    private int m_numEpochs = 1;

    public void setNumEpochs(int numEpochs) { m_numEpochs = numEpochs; }

    @OptionMetadata(description = "Number of epochs", displayName = "numEpochs", displayOrder = 1)
    public int getNumEpochs() { return m_numEpochs; }

    // Number of iterations
    private int m_iterations = 1;

    public void setIterations(int iterations) { m_iterations = iterations; }

    @OptionMetadata(description = "Number of iterations", displayName = "iterations", displayOrder = 1)
    public int getIterations() { return m_iterations; }

    // Truncated backpropagation through time
    private int m_tbpttLength = 0;

    @OptionMetadata(description = "Truncated Backpropagation Through Time length", displayName = "TBPTTLength", displayOrder = 1)
    public int getTBPTTLength() {
        return m_tbpttLength;
    }

    public void setTBPTTLength(int tbpttLength) { m_tbpttLength = tbpttLength; }

    // Learning rate
    private double m_learningRate = 0.01;

    @OptionMetadata(description = "Learning rate", displayName = "learningRate", displayOrder = 1)
    public double getLearningRate() {
        return m_learningRate;
    }

    public void setLearningRate(double learningRate) {
        m_learningRate = learningRate;
    }

    // Momentum (for Nesterov's updater)
    private double m_momentum = 0.9;

    @OptionMetadata(description = "Momentum", displayName = "momentum", displayOrder = 1)
    public double getMomentum() {
        return m_momentum;
    }

    public void setMomentum(double momentum) {
        m_momentum = momentum;
    }

    // Updater
    public Updater m_updater = Updater.NESTEROVS;

    @OptionMetadata(description = "Gradient descent update algorithm to use", displayName = "updater", displayOrder = 1)
    public Updater getUpdater() {
        return m_updater;
    }

    public void setUpdater(Updater updater) {
        m_updater = updater;
    }

    // RMS Decay (for RMSPROP updater)
    public double m_rmsDecay = 0.95;

    @OptionMetadata(description = "RMS decay for RMSPROP updater", displayName = "rmsDecay", displayOrder = 1)
    public double getRMSDecay() {
        return m_rmsDecay;
    }

    public void setRMSDecay(double rmsdecay) {
        m_rmsDecay = rmsdecay;
    }

    // Regularization
    protected boolean m_regularization = false;

    public boolean getRegularization() { return m_regularization; }

    public void setRegularization(boolean regularization) { m_regularization = regularization; }

    // Iterators
    private AbstractDataSetIterator m_iterator = new DefaultDataSetIterator();

    public AbstractDataSetIterator getDataSetIterator() {
        return m_iterator;
    }

    public void setDataSetIterator(AbstractDataSetIterator iterator) {
        m_iterator = iterator;
    }

    // Layer validation
    public void validate() throws Exception {
        if (m_layers.length == 0) {
            throw new Exception("No layers have been added!");
        }
        if( ! (m_layers[ m_layers.length-1 ] instanceof weka.dl4j.layers.RNNOutputLayer) ) {
            throw new Exception("Last layer in RNN network must be an RNN output layer!");
        }
        for (int i = 0; i < m_layers.length - 1; i++) {
            if (!(m_layers[i] instanceof weka.dl4j.layers.LSTM))
                throw new Exception("Currently only LSTM layers for non-output RNN layers are supported");
        }
    }

    /**
     * Get the current number of units (nodes).
     * @param layer
     * @return
     */
    public int getNumUnits(Layer layer) {
        if(layer instanceof LSTM) {
            LSTM tmp = (LSTM) layer;
            return tmp.getNumUnits();
        }
        return -1;
    }

    private int getNumLSTMLayers() {
        int cnt = 0;
        for (int i = 0; i < getLayers().length; i++) {
            if (getLayers()[i].getLayer() instanceof GravesLSTM)
                cnt++;
        }

        return cnt;
    }

    private List<Integer> getLSTMindexes() {
        List<Integer> indexes = new ArrayList<Integer>(getNumLSTMLayers());
        for (int i = 0; i < m_layers.length; i++) {
            if (m_layers[i].getLayer() instanceof GravesLSTM)
                indexes.add(new Integer(i));
        }
        return indexes;
    }

    /*
      Reset LSTM state
     */
    public void clearPreviousState() {
        m_model.rnnClearPreviousState();
    }

    /*
      Set previous state for each LSTM layer. Object must be a list of maps, one for each LSTM layer
     */
    public void setPreviousState(Object previousState){
        List<Map<String, INDArray>> states = ((List<Map<String,INDArray>>) previousState);
        int numLayers = getLayers().length;
        for (int i = 0; i < numLayers; i++) {
            if (getLayers()[i].getLayer() instanceof GravesLSTM)
                m_model.rnnSetPreviousState(i, states.get(i));
        }
    }

    /*
      Get previous state of each LSTM layer
     */
    public List<Map<String, INDArray>> getPreviousState() {
        List<Map<String, INDArray>> states = new ArrayList<Map<String, INDArray>>(getNumLSTMLayers());
        int numLayers = getLayers().length;
        for (int i = 0; i < numLayers; i ++) {
            if (getLayers()[i].getLayer() instanceof GravesLSTM)
                 states.add(m_model.rnnGetPreviousState(i));
        }
        return states;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        // validate
        validate();
        m_zeroR = null;
        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();
        // classifier defaults to Zero Rule
        if (data.numInstances() == 0 || data.numAttributes() < 2) {
            m_zeroR.buildClassifier(data);
            return;
        }
        // can classifier handle the data?
        getCapabilities().testWithFail(data);
        // remove missing data
        m_replaceMissing = new ReplaceMissingValues();
        m_replaceMissing.setInputFormat(data);
        data = Filter.useFilter(data, m_replaceMissing);
        if (m_standardizeInsteadOfNormalize) {
            m_normalize = new Standardize();
            // we want to also normalize the class
            m_normalize.setOptions(new String[] { "-unset-class-temporarily" } );
        } else {
            m_normalize = new Normalize();
            m_normalize.setOptions(new String[] { "-unset-class-temporarily" } );
        }
        m_normalize.setInputFormat(data);
        data = Filter.useFilter(data, m_normalize);
//        m_nominalToBinary = new NominalToBinary();
//        m_nominalToBinary.setInputFormat(data);
//        data = Filter.useFilter(data, m_nominalToBinary);


        DataSet trainingData = RNNinstancesToDataSet(data);
        if(getDebug()) {
        	System.err.println(trainingData);
        }

        // construct the rnn configuration
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        builder.seed(getSeed());
        builder.biasInit(0);
        builder.miniBatch(true);
        builder.iterations(getIterations());
        builder.learningRate(getLearningRate());
        builder.momentum(getMomentum());
        builder.gradientNormalization(getGradientNormalization());
        builder.gradientNormalizationThreshold(getGradientNormalizationThres());
        builder.optimizationAlgo(getOptimizationAlgorithm());
        builder.updater(getUpdater());
        builder.rmsDecay(getRMSDecay());
        builder.regularization(getRegularization());

        NeuralNetConfiguration.ListBuilder listBuilder = builder.list();

        if(getTBPTTLength() != 0) {
            int tbpttLength = getTBPTTLength();
            listBuilder.backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength);
        }
        listBuilder = listBuilder.pretrain(false).backprop(true);

        int numInputAttributes = getDataSetIterator().getNumAttributes(data);

        // set layers
        for (int x = 0; x < m_layers.length; x++) {
            // output layer
            if ( x == m_layers.length-1 ) {
                // if this is the only layer
                if( x == 0)
                    m_layers[x].setNumIncoming(numInputAttributes);
                else // otherwise
                    m_layers[x].setNumIncoming( getNumUnits(m_layers[x-1]) );
                // data.numClasses() is the number of class labels, not the number of possible values for one class label
                // so the number of output nodes of the NN is the number of class labels
                m_layers[x].setNumOutgoing(data.numClasses());
                listBuilder = listBuilder.layer(x, m_layers[x].getLayer());
            // input layer
            } else if (x == 0) {
                m_layers[x].setNumIncoming(numInputAttributes);
                listBuilder = listBuilder.layer(x, m_layers[x].getLayer() );
            // neither input nor output layer
            } else {
                m_layers[x].setNumIncoming( getNumUnits(m_layers[x-1]) );
                listBuilder = listBuilder.layer(x, m_layers[x].getLayer() );
            }
        }

        MultiLayerConfiguration conf = listBuilder.build();
        // build the network
        m_model = new MultiLayerNetwork(conf);
        m_model.init();

        if( getVisualisation() ) {
            m_model.setListeners(new HistogramIterationListener(1));
        }

        if( getDebug() ) {
            System.err.println( conf.toJson() );
        }

        // Train the model with m_numEpochs complete passes through the data set
        for(int i = 0; i < getNumEpochs(); i++) {
            if( getDebug() )
                System.err.println("**** Epoch " + i + " ****");
            m_model.fit(trainingData);
        }
        // Make net "predict" the known values so it stores the state it's in for making the next prediction
        // using rnnTimeStep. When starting predictions we can pick up from last known output
//        INDArray trainMatrix = trainingData.getFeatureMatrix();
//        INDArray output = m_model.rnnTimeStep(trainMatrix);

        if( getDebug() ) {
//            System.out.println("\ntraining data model output: " + output);
            System.out.println("\n**************** TRAINING DONE ****************\n");
        }
    }

    public double[] distributionForInstance(Instance inst) throws Exception {
        if (m_zeroR != null) {
            return m_zeroR.distributionForInstance(inst);
        }
        // Transform the input instance
        m_normalize.input(inst);
        inst = m_normalize.output();

        Instances insts = new Instances( inst.dataset() );
        insts.add(inst);

        DataSet testData = RNNinstancesToDataSet(insts);
        INDArray testMatrix = testData.getFeatureMatrix();
        System.out.println("Test feature matrix: " + testMatrix);

        // For RNNs the state of the network is important (namely all the LSTM layers). After training it with the
        // given data, the prediction it makes for the next time step must take all the past into account. Of course
        // it would be rather unefficient for it to go all the way back to the beginning of the time series each time
        // made a prediction, so we use rnnTimeStep to grab the current network state (which knows what the last output
        // was) and make the next prediction only based on the last time step.
        INDArray predicted = m_model.rnnTimeStep(testMatrix);
        // using output() would require us to input to the test matrix all the past instances
        // with rnnTimeStep() we can just input the last instance, as it has stored the state from the last prediction
        // so conceptually all the past data is still always used
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
        // gradient norm
        result.add("-" + Constants.GRADIENT_NORM);
        result.add("" + getGradientNormalization().name());
        // gradient norm threshold
        result.add("-" + Constants.GRADIENT_NORM_THRESHOLD);
        result.add("" + getGradientNormalizationThres());
        // optimization algorithm
        result.add("-" + Constants.OPTIMIZATION_ALGORITHM);
        result.add("" + getOptimizationAlgorithm().name());
        // tbptt length
        result.add("-" + Constants.TBPTT_LENGTH);
        result.add("" + getTBPTTLength());
        // number of iterations
        result.add("-" + Constants.ITERATIONS);
        result.add("" + getIterations());
        // number of epochs
        result.add("-" + Constants.NUM_EPOCHS);
        result.add("" + getNumEpochs());
        // learning rate
        result.add("-" + Constants.LEARNING_RATE);
        result.add("" + getLearningRate());
        // momentum
        result.add("-" + Constants.MOMENTUM);
        result.add("" + getMomentum());
        // updater
        result.add("-" + Constants.UPDATER);
        result.add("" + getUpdater().name());
        // rms decay
        result.add("-" + Constants.RMS_DECAY);
        result.add("" + getRMSDecay());
        // regularization
        result.add("-" + Constants.REGULARIZATION);
        result.add("" + getRegularization());
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
            layers.add(new weka.dl4j.layers.LSTM());
        }
        setLayers(layers.toArray(new weka.dl4j.layers.Layer[layers.size()]));

        //.batchSize( getTrainBatchSize() )

        // optimization algorithm
        String tmp = weka.core.Utils.getOption(Constants.OPTIMIZATION_ALGORITHM, options);
        if(!tmp.equals("")) setOptimizationAlgorithm( OptimizationAlgorithm.valueOf(tmp) );
        // TBPTT length
        tmp = weka.core.Utils.getOption(Constants.TBPTT_LENGTH, options);
        if(!tmp.equals("")) setTBPTTLength( Integer.parseInt(tmp) );
        // number of iterations
        tmp = weka.core.Utils.getOption(Constants.ITERATIONS, options);
        if(!tmp.equals("")) setIterations( Integer.parseInt(tmp) );
        // gradient normalization
        tmp = weka.core.Utils.getOption(Constants.GRADIENT_NORM, options);
        if(!tmp.equals("")) setGradientNormalization( GradientNormalization.valueOf(tmp) );
        // gradient normalization threshold
        tmp = weka.core.Utils.getOption(Constants.GRADIENT_NORM_THRESHOLD, options);
        if(!tmp.equals("")) setGradientNormalizationThres( Double.parseDouble(tmp) );
        // number of epochs
        tmp = weka.core.Utils.getOption(Constants.NUM_EPOCHS, options);
        if(!tmp.equals("")) setNumEpochs( Integer.parseInt(tmp) );
        // learning rate
        tmp = weka.core.Utils.getOption(Constants.LEARNING_RATE, options);
        if(!tmp.equals("")) setLearningRate( Double.parseDouble(tmp) );
        // momentum
        tmp = weka.core.Utils.getOption(Constants.MOMENTUM, options);
        if(!tmp.equals("")) setMomentum( Double.parseDouble(tmp) );
        // updater
        tmp = weka.core.Utils.getOption(Constants.UPDATER, options);
        if(!tmp.equals("")) setUpdater( Updater.valueOf(tmp) );
        // rms decay
        tmp = weka.core.Utils.getOption(Constants.RMS_DECAY, options);
        if(!tmp.equals("")) setRMSDecay( Double.parseDouble(tmp) );
        // regularization
        tmp = weka.core.Utils.getOption(Constants.REGULARIZATION, options);
        if(!tmp.equals("")) setRegularization( Boolean.valueOf(tmp) );
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
                "\tList of layers that define the RNN",
                "-layers",
                1,
                "-layers <layer arguments>"
        ));
        // optim alg
        String optNames = "(";
        for(int i = 0; i < OptimizationAlgorithm.values().length; i++) {
            optNames += ( OptimizationAlgorithm.values()[i].toString() + " | ");
        }
        optNames += ")";
        v.add(new Option(
                "\tOptimization algorithm to use",
                "-optim",
                1,
                "-optim " + optNames
        ));
        // tbptt length
        v.add(new Option(
                "\ttbptt length",
                "-tl",
                1,
                "-tl <int>"
        ));
        // number of iterations
        v.add(new Option(
                "\tNumber of iterations",
                "-nI",
                1,
                "-nI <int>"
        ));
        // learning rate
        v.add(new Option(
                "\tLearning rate",
                "-lr",
                1,
                "-lr <float>"
        ));
        // num epochs
        v.add(new Option(
                "\tNumber of epochs",
                "-nE",
                1,
                "-nE <int>"
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
        // rms decay
        v.add(new Option(
                "\t RMS decay",
                "-rms_decay",
                1,
                "-rms_decay <double>"
        ));
        // regularization
        optNames = "(";
            optNames += "TRUE | FALSE | ";
        optNames += ")";
        v.add(new Option(
                "\tWether to use regularization methods",
                "-regularization",
                1,
                "-regularization " + optNames
        ));
        // gradient normalization
        optNames = "(";
        for(int i = 0; i < GradientNormalization.values().length; i++) {
            optNames += ( GradientNormalization.values()[i].toString() + " | ");
        }
        optNames += ")";
        v.add(new Option(
                "\tGradient normalization algorithm to use",
                "-grad_norm",
                1,
                "-grad_norm " + optNames
        ));
        // gradient normalization threshold
        v.add(new Option(
                "\tGradient normalization threshold",
                "-grad_norm_thres",
                1,
                "-grad_norm_thres <double>"
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
