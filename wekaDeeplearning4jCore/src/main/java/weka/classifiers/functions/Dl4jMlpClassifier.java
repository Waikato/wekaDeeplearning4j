/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    Dl4jMlpClassifier.java
 *    Copyright (C) 2016 University of Waikato, Hamilton, New Zealand
 *
 */
package weka.classifiers.functions;

import java.io.*;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.output.CountingOutputStream;
import org.apache.commons.io.output.NullOutputStream;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;

import weka.classifiers.IterativeClassifier;
import weka.classifiers.RandomizableClassifier;
import weka.classifiers.rules.ZeroR;
import weka.core.*;
import weka.dl4j.earlystopping.EarlyStopping;
import weka.dl4j.iterators.instance.*;
import weka.dl4j.layers.DenseLayer;
import weka.dl4j.listener.EpochListener;
import weka.dl4j.listener.FileIterationListener;
import org.deeplearning4j.nn.conf.layers.Layer;
import weka.dl4j.layers.OutputLayer;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.zoo.EmptyNet;
import weka.dl4j.zoo.FaceNetNN4Small2;
import weka.dl4j.zoo.GoogLeNet;
import weka.dl4j.zoo.ZooModel;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import javax.naming.OperationNotSupportedException;

/**
 * A wrapper for DeepLearning4j that can be used to train a multi-layer
 * perceptron using that library.
 *
 * @author Christopher Beckham
 * @author Eibe Frank
 *
 * @version $Revision: 11711 $
 */
@Slf4j
public class Dl4jMlpClassifier extends RandomizableClassifier implements
  BatchPredictor, CapabilitiesHandler, IterativeClassifier {

  /** The ID used for serializing this class. */
  protected static final long serialVersionUID = -6363254116597574265L;

  /** Filter used to replace missing values. */
  protected ReplaceMissingValues m_replaceMissing;

  /** Filter used to normalize or standardize the data. */
  protected Filter m_Filter;

  /** Filter used to convert nominal attributes to binary numeric attributes. */
  protected NominalToBinary m_nominalToBinary;

  /**
   * ZeroR classifier, just in case we don't actually have any data to train a
   * network.
   */
  protected ZeroR m_zeroR;


  /** The actual neural network model. **/
  protected transient ComputationGraph m_model;

  /** The model zoo model. **/
  protected ZooModel m_zooModel = new EmptyNet();

  /** The size of the serialized network model in bytes. **/
  protected long m_modelSize;

  /** The file that log information will be written to. */
  protected File m_logFile = new File(Paths.get(System.getenv("WEKA_HOME"), "network.log").toString());

  /** The layers of the network. */
  protected Layer[] m_layers = new Layer[] {new OutputLayer()};

  /** The configuration of the network. */
  protected NeuralNetConfiguration m_netConfig = new NeuralNetConfiguration();

  /** The configuration for early stopping. */
  protected EarlyStopping m_earlyStopping = new EarlyStopping();

  /** The number of epochs to perform. */
  protected int m_numEpochs = 10;

  /** The number of epochs that have been performed. */
  protected int m_NumEpochsPerformed;

  /** The dataset trainIterator. */
  protected transient DataSetIterator m_trainIterator;

  /** The dataset validationIterator. */
  protected transient DataSetIterator m_valIterator;

  /** The training instances (set to null when done() is called). */
  protected Instances m_trainData;

  /** The validation instances (set to null when done() is called). */
  protected Instances m_valData;

  /** The instance iterator to use. */
  protected AbstractInstanceIterator m_instanceIterator = new DefaultInstanceIterator();

  /** Queue size for AsyncDataSetIterator (if < 1, AsyncDataSetIterator is not used) */
  protected int m_queueSize = 0;

  /** filter: Normalize training data */
  public static final int FILTER_NORMALIZE = 0;
  /** filter: Standardize training data */
  public static final int FILTER_STANDARDIZE = 1;
  /** filter: No normalization/standardization */
  public static final int FILTER_NONE = 2;

  /** The filter to apply to the training data */
  public static final Tag [] TAGS_FILTER = {
          new Tag(FILTER_NORMALIZE, "Normalize training data"),
          new Tag(FILTER_STANDARDIZE, "Standardize training data"),
          new Tag(FILTER_NONE, "No normalization/standardization"),
  };

  /** Whether to normalize/standardize/neither */
  protected int m_filterType = FILTER_STANDARDIZE;

  /** Coefficients used for normalizing the class */
  protected double m_x1 = 1.0;
  protected double m_x0 = 0.0;


  /**
   * Training listener list
   */
  private IterationListener m_iterationListener = new EpochListener();

  /**
   * The main method for running this class.
   *
   * @param argv the command-line arguments
   */
  public static void main(String[] argv) {
    runClassifier(new Dl4jMlpClassifier(), argv);
  }

  public String globalInfo() {
    return "Classification and regression with multilayer perceptrons using DeepLearning4J.";
  }

  /**
   * Returns default capabilities of the classifier.
   *
   * @return the capabilities of this classifier
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    if (getInstanceIterator() instanceof ImageInstanceIterator) {
      result.enable(Capabilities.Capability.STRING_ATTRIBUTES);
    } else {
      result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
      result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
      result.enable(Capabilities.Capability.DATE_ATTRIBUTES);
      result.enable(Capabilities.Capability.MISSING_VALUES);
      result.enableDependency(Capabilities.Capability.STRING_ATTRIBUTES); // User might switch to ImageDSI in GUI
    }

    // class
    result.enable(Capabilities.Capability.NOMINAL_CLASS);
    result.enable(Capabilities.Capability.NUMERIC_CLASS);
    result.enable(Capabilities.Capability.DATE_CLASS);
    result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

    return result;
  }

  /**
   * Custom serialization method.
   *
   * @param oos the object output stream
   * @throws IOException
   */
  private void writeObject(ObjectOutputStream oos) throws IOException {

    // figure out size of the written network
    CountingOutputStream cos = new CountingOutputStream(new NullOutputStream());
    if (m_replaceMissing != null) {
      ModelSerializer.writeModel(m_model, cos, false);
    }
    m_modelSize = cos.getByteCount();

    // default serialization
    oos.defaultWriteObject();

    // actually write the network
    if (m_replaceMissing != null) {
      ModelSerializer.writeModel(m_model, oos, false);
    }
  }

  /**
   * Custom deserialization method
   *
   * @param ois the object input stream
   * @throws ClassNotFoundException
   * @throws IOException
   */
  private void readObject(ObjectInputStream ois) throws ClassNotFoundException, IOException {
    ClassLoader origLoader = Thread.currentThread().getContextClassLoader();
    try {
      Thread.currentThread().setContextClassLoader(this.getClass().getClassLoader());
      // default deserialization
      ois.defaultReadObject();

      // restore the network model
      if (m_replaceMissing != null) {
        File tmpFile = File.createTempFile("restore", "multiLayer");
        tmpFile.deleteOnExit();
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(tmpFile));
        long remaining = m_modelSize;
        while (remaining > 0) {
          int bsize = 10024;
          if (remaining < 10024) {
            bsize = (int)remaining;
          }
          byte[] buffer = new byte[bsize];
          int len = ois.read(buffer);
          if (len == -1) {
            throw new IOException("Reached end of network model prematurely during deserialization.");
          }
          bos.write(buffer, 0, len);
          remaining -= len;
        }
        bos.flush();
        m_model = ModelSerializer.restoreComputationGraph(tmpFile, false);
      }
    } finally {
      Thread.currentThread().setContextClassLoader(origLoader);
    }
  }

  /**
   * Get the log file
   *
   * @return the log file
   */
  public File getLogFile() {
    return m_logFile;
  }

  /**
   * Set the log file
   *
   * @param logFile the log file
   */
  @OptionMetadata(displayName = "log file",
    description = "The name of the log file to write loss information to "
      + "(default = $WEKA_HOME/network.log).", commandLineParamName = "logFile",
    commandLineParamSynopsis = "-logFile <string>", displayOrder = 1)
  public void setLogFile(File logFile) {
    m_logFile = logFile;
  }

  public Layer[] getLayers() {
    return m_layers;
  }

  @OptionMetadata(displayName = "layer specification.",
    description = "The specification of a layer. This option can be used multiple times.",
    commandLineParamName = "layer",
    commandLineParamSynopsis = "-layer <string>", displayOrder = 2)
  public void setLayers(Layer[] layers) {

    layers = fixDuplicateLayerNames(layers);
    m_layers = layers;
  }

  /**
   * Check if layer names are duplicate. If so, correct them by appending indices
   * @param layers Array of network layer
   * @return Array of network layer with corrected names
   */
  private Layer[] fixDuplicateLayerNames(Layer[] layers) {
    Set<String> names = Arrays.stream(layers).map(Layer::getLayerName).collect(Collectors.toSet());

    for (String name : names){
      // Find duplicates with the same name
      List<Layer> duplicates = Arrays.stream(layers)
              .filter(l -> name.equals(l.getLayerName()))
              .collect(Collectors.toList());

      // If no duplicates were found, continue
      if (duplicates.size() == 1){
        continue;
      }

      // For each duplicate add an index
      for (int i = 0; i < duplicates.size(); i++){
        duplicates.get(i).setLayerName(name + " " + (i + 1));
      }

    }
    return layers;
  }

  public int getNumEpochs() {
    return m_numEpochs;
  }

  @OptionMetadata(description = "The number of epochs to perform.",
    displayName = "number of epochs", commandLineParamName = "numEpochs",
    commandLineParamSynopsis = "-numEpochs <int>", displayOrder = 4)
  public void setNumEpochs(int numEpochs) {
    m_numEpochs = numEpochs;
  }

  @OptionMetadata(description = "The instance trainIterator to use.",
          displayName = "instance iterator", commandLineParamName = "iterator",
          commandLineParamSynopsis = "-iterator <string>", displayOrder = 6)
  public AbstractInstanceIterator getInstanceIterator() {
    return m_instanceIterator;
  }

  public void setInstanceIterator(AbstractInstanceIterator iterator) {
    m_instanceIterator = iterator;
  }

  @OptionMetadata(description = "The neural network configuration to use.",
          displayName = "network configuration", commandLineParamName = "config",
          commandLineParamSynopsis = "-config <string>", displayOrder = 7)
  public NeuralNetConfiguration getNeuralNetConfiguration() {
    return m_netConfig;
  }

  public void setNeuralNetConfiguration(NeuralNetConfiguration config) {
    if(!(m_zooModel instanceof EmptyNet)){
      log.warn("Custom NeuralNetConfiguration was set while a ZooModel has been set." +
              "This has no effect.");
    }
    m_netConfig = config;
  }

  @OptionMetadata(description = "The early stopping configuration to use.",
          displayName = "early stopping configuration", commandLineParamName = "early-stopping",
          commandLineParamSynopsis = "-early-stopping <string>", displayOrder = 7)
  public EarlyStopping getEarlyStopping() {
    return m_earlyStopping;
  }

  public void setEarlyStopping(EarlyStopping config) {
    m_earlyStopping = config;
  }

  @OptionMetadata(description = "The type of normalization to perform.",
          displayName = "attribute normalization", commandLineParamName = "normalization",
          commandLineParamSynopsis = "-normalization <int>", displayOrder = 8)
  public SelectedTag getFilterType() {
    return new SelectedTag(m_filterType, TAGS_FILTER);
  }

  public void setFilterType(SelectedTag newType) {
    if (newType.getTags() == TAGS_FILTER) {
      m_filterType = newType.getSelectedTag().getID();
    }
  }

  public int getQueueSize() {
    return m_queueSize;
  }

  @OptionMetadata(description = "The queue size for asynchronous data transfer (default: 0, synchronous transfer).",
          displayName = "queue size for asynchronous data transfer", commandLineParamName = "queueSize",
          commandLineParamSynopsis = "-queueSize <int>", displayOrder = 9)
  public void setQueueSize(int QueueSize) {
    m_queueSize = QueueSize;
  }

  /**
   * The method used to train the classifier.
   *
   * @param data set of instances serving as training data
   * @throws Exception if something goes wrong in the training process
   */
  @Override
  public void buildClassifier(Instances data) throws Exception {

    // Initialize classifier
    initializeClassifier(data);


    boolean cont = true;
    while (cont) {
      // Next epoch
      cont = next();
    }

    // Clean up
    done();
  }

  /**
   * The method used to initialize the classifier.
   *
   * @param data set of instances serving as training data
   * @throws Exception if something goes wrong in the training process
   */
  @Override
  public void initializeClassifier(Instances data) throws Exception {

    // Can classifier handle the data?
    getCapabilities().testWithFail(data);

    // Check basic network structure
    if (m_layers.length == 0) {
      throw new Exception("No layers have been added!");
    }

    if (!(m_layers[m_layers.length - 1] instanceof OutputLayer)) {
      throw new Exception("Last layer in network must be an output layer!");
    }

    // Apply preprocessing
    data = preProcessInput(data);

    // Split train/validation
    double valSplit = m_earlyStopping.getValidationSetPercentage();
    Instances trainData = null;
    Instances valData = null;
    if (useEarlyStopping()){
      Instances[] insts = splitTrainVal(data, valSplit);
      trainData = insts[0];
      valData = insts[1];
      validateSplit(trainData, valData);
    } else {
      trainData = data;
    }


    if (trainData == null) {
      return;
    } else {
      m_trainData = trainData;
      m_valData = valData;
    }

    ClassLoader origLoader = Thread.currentThread().getContextClassLoader();
    try {
      Thread.currentThread().setContextClassLoader(this.getClass().getClassLoader());

      // If zoo model was set, use this model as internal MultiLayerNetwork
      if(!(m_zooModel instanceof EmptyNet)){
        createZooModel();
      } else {
        createModel();
      }
      // Setup the datasetiterators (needs to be done after the model initialization)
      m_trainIterator = getDataSetIterator(m_trainData);

      if (useEarlyStopping()){
        m_valIterator = getDataSetIterator(m_valData);
      }

      // Print model architecture
      if (getDebug()) {
        log.info(m_model.conf().toYaml());
      }


      // Set the iteration listener
      m_model.setListeners(getListener());
      m_earlyStopping.init(m_valIterator);

      m_NumEpochsPerformed = 0;
    } finally {
      Thread.currentThread().setContextClassLoader(origLoader);
    }
  }

  private void validateSplit(Instances trainData, Instances valData) throws WekaException {
    if (m_earlyStopping.getValidationSetPercentage() < 10e-8){
      // Use no validation set at all
      return;
    }
    int classIndex = trainData.classIndex();
    int valDataNumDinstinctClassValues = valData.numDistinctValues(classIndex);
    int trainDataNumDistinctClassValues = trainData.numDistinctValues(classIndex);
    if(trainData.numClasses() > 1 && valDataNumDinstinctClassValues != trainDataNumDistinctClassValues){
      throw new WekaException("The validation data did not contain the same classes as the training data. " +
              "You should increase the validation split.");
    }
  }

  /**
   * Split the dataset into p% traind an (100-p)% test set
   *
   * @param data Input data
   * @param p    train percentage
   * @return Array of instances: (0) Train, (1) Test
   * @throws Exception Filterapplication went wrong
   */
  public static Instances[] splitTrainVal(Instances data, double p) throws Exception {
    // Validate percentage
//    if (!(0 < p && p < 100)){
//      throw new WekaException("Validation split size must be in 0 < p < 100.");
//    }

    Randomize rand = new Randomize();
    rand.setInputFormat(data);
    rand.setRandomSeed(42);
    data = Filter.useFilter(data, rand);

    RemovePercentage rp = new RemovePercentage();
    rp.setInputFormat(data);
    rp.setPercentage(p);
    Instances train = Filter.useFilter(data, rp);


    rp = new RemovePercentage();
    rp.setInputFormat(data);
    rp.setPercentage(p);
    rp.setInvertSelection(true);
    Instances test = Filter.useFilter(data, rp);

    return new Instances[]{train, test};
  }

  /**
   * Generates a DataSetIterator based on the given instances.
   * @param data Input instances
   * @return DataSetIterator
   * @throws Exception
   */
  private DataSetIterator getDataSetIterator(Instances data) throws Exception {
    DataSetIterator it = m_instanceIterator.getDataSetIterator(data, getSeed());
    if (m_queueSize > 0) {
      it = new AsyncDataSetIterator(it, m_queueSize);
    }
    return it;
  }

  /**
   * Apply weka filter preprocessing to the input data.
   * @param data Data as weka instances
   * @return Preprocessed instances
   * @throws Exception Preprocessing failed
   */
  private Instances preProcessInput(Instances data) throws Exception {
    // Remove instances with missing class and check that instances and
    // predictor attributes remain.
    data = new Instances(data);
    data.deleteWithMissingClass();
    m_zeroR = null;
    int numSamples = data.numInstances();
    if (numSamples == 0 || data.numAttributes() < 2) {
      m_zeroR = new ZeroR();
      m_zeroR.buildClassifier(data);
      return null;
    }

    // Retrieve two different class values used to determine filter
    // transformation
    double y0 = data.instance(0).classValue();
    int index = 1;
    while (index < numSamples
            && data.instance(index).classValue() == y0) {
      index++;
    }
    if (index == numSamples) {
      // degenerate case, all class values are equal
      // we don't want to deal with this, too much hassle
      throw new Exception(
              "All class values are the same. At least two class values should be different");
    }
    double y1 = data.instance(index).classValue();

    // Init and apply the filters
    data = initFilters(data);

    double z0 = data.instance(0).classValue();
    double z1 = data.instance(index).classValue();
    m_x1 = (y0 - y1) / (z0 - z1); // no division by zero, since y0 != y1
    // guaranteed => z0 != z1 ???
    m_x0 = (y0 - m_x1 * z0); // = y1 - m_x1 * z1

    // Randomize the data, just in case
    Random rand = new Random(getSeed());
    data.randomize(rand);
    return data;
  }

  /**
   * Initialize {@link ReplaceMissingValues}, {@link NominalToBinary} and {@link Standardize} or {@link Normalize}
   * filters
   * @param data Input data to set the input formal of the filters
   * @return Transformed data
   * @throws Exception Filter can not be initialized
   */
  private Instances initFilters(Instances data) throws Exception {
    // Replace missing values
    m_replaceMissing = new ReplaceMissingValues();
    m_replaceMissing.setInputFormat(data);
    data = Filter.useFilter(data, m_replaceMissing);

    // Replace nominal attributes by binary numeric attributes.
    m_nominalToBinary = new NominalToBinary();
    m_nominalToBinary.setInputFormat(data);
    data = Filter.useFilter(data, m_nominalToBinary);

    // Standardize or normalize (as requested), including the class

    if (m_filterType == FILTER_STANDARDIZE) {
      m_Filter = new Standardize();
      m_Filter.setOptions(new String[]{"-unset-class-temporarily"});
      m_Filter.setInputFormat(data);
      data = Filter.useFilter(data, m_Filter);
    } else if (m_filterType == FILTER_NORMALIZE) {
      m_Filter = new Normalize();
      m_Filter.setOptions(new String[]{"-unset-class-temporarily"});
      m_Filter.setInputFormat(data);
      data = Filter.useFilter(data, m_Filter);
    } else {
      m_Filter = null;
    }

    return data;
  }

  /**
   * Build the Zoomodel instance
   * @return ComputationGraph instance
   * @throws WekaException Either the .init operation on the current zooModel was not supported or the data shape does
   * not fit the chosen zooModel
   */
  private void createZooModel() throws WekaException {
      ImageInstanceIterator iii = (ImageInstanceIterator) getInstanceIterator();
      int newWidth = iii.getWidth();
      int newHeight = iii.getHeight();
      int channels = iii.getNumChannels();
      boolean initSuccessful = false;
      while (!initSuccessful){
        // Increase width and height
        int[] newShape =  new int[]{channels, newHeight, newWidth};
        int[][] shapeWrap = new int[][]{newShape};
        setInstanceIterator(new ResizeImageInstanceIterator(iii, newWidth, newHeight));
        initSuccessful = initZooModel(m_trainData.numClasses(), getSeed(), shapeWrap);

        newWidth *= 1.2;
        newHeight *= 1.2;
        if (!initSuccessful){
          log.warn("The shape of the data did not fit the chosen " +
                  "model. It was therefore resized to ({}x{}x{}).",
                  channels, newHeight, newWidth);
        }
      }

  }

  private boolean initZooModel(int numClasses, long seed, int[][] newShape) {
    try {
      m_model = m_zooModel.init(numClasses, seed, newShape);
      return true;
    } catch (OperationNotSupportedException e) {
      throw new RuntimeException("ZooModel was not set, but createZooModel could be called. Invalid situation", e);
    } catch (DL4JInvalidConfigException | DL4JInvalidInputException e) {
      return false;
    }
  }

  /**
   * Build the multilayer network defined by the networkconfiguration and the list of layers.
   * @throws Exception
   */
  private void createModel() throws Exception {
    final INDArray features = getDataSetIterator(m_trainData).next().getFeatures();
    ComputationGraphConfiguration.GraphBuilder gb = m_netConfig.builder()
            .seed(getSeed())
            .graphBuilder();


    // Set ouput size
    ((OutputLayer)m_layers[m_layers.length-1]).setNOut(m_trainData.numClasses());

    String currentInput = "input";
    gb.addInputs(currentInput);
    // Collect layers
    for (Layer m_layer : m_layers) {
      String lName = m_layer.getLayerName();
      gb.addLayer(lName, m_layer, currentInput);
      currentInput = lName;
    }
    gb.setOutputs(currentInput);
    gb.setInputTypes(InputType.inferInputTypes(features));



    // Set input type for the first layer manually since the builder above does not overwrite
    // the input type. This is especially problematic in the Weka AbstractTest since no new CLF instance
    // is created on new tests and therefor buildClassifier is called multiple times.
    boolean override = true;
    Layer inputLayer = m_layers[0];
    if (getInstanceIterator() instanceof ImageInstanceIterator
            && (inputLayer instanceof DenseLayer || inputLayer instanceof OutputLayer)){
      ImageInstanceIterator iii = ((ImageInstanceIterator)getInstanceIterator());
      int height = iii.getHeight();
      int width = iii.getWidth();
      int channels = iii.getNumChannels();
      inputLayer.setNIn(InputType.convolutionalFlat(height, width, channels), override);
    } else if (getInstanceIterator() instanceof ConvolutionInstanceIterator && inputLayer instanceof DenseLayer){
      ConvolutionInstanceIterator iii = ((ConvolutionInstanceIterator)getInstanceIterator());
      int height = iii.getHeight();
      int width = iii.getWidth();
      int channels = iii.getNumChannels();
      inputLayer.setNIn(InputType.convolutionalFlat(height, width, channels), override);
    } else {
      inputLayer.setNIn(InputType.inferInputType(features), override);
    }

    ComputationGraphConfiguration conf = gb.pretrain(false).backprop(true).build();
    ComputationGraph model = new ComputationGraph(conf);
    model.init();
    m_model = model;
  }

  /**
   * Get the iterationlistener
   * @throws Exception
   */
  private List<IterationListener> getListener() throws Exception {
    int numSamples = m_trainData.numInstances();
    List<IterationListener> listeners = new ArrayList<>();

    // Initialize weka listener
    if (m_iterationListener instanceof weka.dl4j.listener.IterationListener) {
      int numEpochs = getNumEpochs();
      ((weka.dl4j.listener.IterationListener) m_iterationListener).init
              (m_trainData.numClasses(), numEpochs, numSamples, m_trainIterator, m_valIterator);
      listeners.add(m_iterationListener);
    }


    // if the log file doesn't point to a directory, set up the listener
    if (getLogFile() != null && !getLogFile().isDirectory()) {
      FileIterationListener fil = new FileIterationListener(getLogFile().getAbsolutePath());
      fil.init(m_trainData.numClasses(), getNumEpochs(), numSamples, m_trainIterator, m_valIterator);
      listeners.add(fil);
    }

    return listeners;
  }

  /**
   * Perform another epoch.
   */
  public boolean next() throws Exception {

    if (m_NumEpochsPerformed >= getNumEpochs() || m_zeroR != null || m_trainData == null) {
      return false;
    }

    ClassLoader origLoader = Thread.currentThread().getContextClassLoader();
    try {
      Thread.currentThread().setContextClassLoader(this.getClass().getClassLoader());
      m_model.fit(m_trainIterator); // Note that this calls the reset() method of the trainIterator
      if (getDebug()) {
        log.info("*** Completed epoch {} ***", m_NumEpochsPerformed + 1);
      }
      m_NumEpochsPerformed++;
    } finally {
      Thread.currentThread().setContextClassLoader(origLoader);
    }

    // Evaluate early stopping
    if (useEarlyStopping()){
      boolean continueTraining = m_earlyStopping.evaluate(m_model);
      if (!continueTraining){
        log.info("Early stopping has stopped the training process. The " +
                        "validation has not improved anymore after {} epochs. Training " +
                        "finished.",
                m_earlyStopping.getMaxEpochsNoImprovement());
      }
      return continueTraining;
    }


    return true;
  }

  /**
   * Use early stopping only if valid split percentage
   * @return True if early stopping should be used
   */
  public boolean useEarlyStopping(){
    double p = m_earlyStopping.getValidationSetPercentage();
    return 0 < p && p < 100;
  }

  /**
   * Clean up after learning.
   */
  public void done() {

    m_trainData = null;
  }

  /**
   * Set the modelzoo zooModel
   *
   * @param zooModel The predefined zooModel
   */
  @OptionMetadata(displayName = "zooModel",
          description = "The model-architecture to choose from the modelzoo " +
                  "(default = no model).", commandLineParamName = "zooModel",
          commandLineParamSynopsis = "-zooModel <string>", displayOrder = 11)
  public void setZooModel(ZooModel zooModel){


    if (zooModel instanceof GoogLeNet || zooModel instanceof FaceNetNN4Small2){
      throw new RuntimeException("The zoomodel you have selected is currently" +
              " not supported! Please select another one.");
    }

    m_zooModel = zooModel;
  }

  /**
   * Get the modelzoo model
   * @return The modelzoo model object
   */
  public ZooModel getZooModel(){
    return m_zooModel;
  }

  @OptionMetadata(displayName = "set the iteration listener",
          description = "Set the iteration listener.", commandLineParamName = "iteration-listener",
          commandLineParamSynopsis = "-iteration-listener <string>", displayOrder = 12)
  public void setIterationListener(IterationListener l) {
    m_iterationListener = l;
  }

  public IterationListener getIterationListener() {
    return m_iterationListener;
  }

  /**
   * Performs efficient batch prediction
   *
   * @return true, as LogitBoost can perform efficient batch prediction
   */
  @Override
  public boolean implementsMoreEfficientBatchPrediction() {
    return true;
  }


  /**
   * The method to use when making a prediction for a test instance. Use distributionsForInstances() instead
   * for speed if possible.
   *
   * @param inst the instance to get a prediction for
   * @return the class probability estimates (if the class is nominal) or the
   *         numeric prediction (if it is numeric)
   * @throws Exception if something goes wrong at prediction time
   */
  @Override
  public double[] distributionForInstance(Instance inst) throws Exception {

    Instances data = new Instances(inst.dataset());
    data.add(inst);
    return distributionsForInstances(data)[0];
  }

  /**
   * The method to use when making predictions for test instances.
   *
   * @param insts the instances to get predictions for
   * @return the class probability estimates (if the class is nominal) or the
   *         numeric predictions (if it is numeric)
   * @throws Exception if something goes wrong at prediction time
   */
  @Override
  public double[][] distributionsForInstances(Instances insts) throws Exception {

    // Do we only have a ZeroR model?
    if (m_zeroR != null) {
      return m_zeroR.distributionsForInstances(insts);
    }

    // Process input data to have the same filters applied as the training data
    insts = applyFilters(insts);


    // Get predictions
    final DataSetIterator it = getDataSetIterator(insts);
    double[][] preds = new double[insts.numInstances()][insts.numClasses()];

    int offset = 0;
    boolean next = true;
    // Get predictions batch-wise
    while (next){
      INDArray predBatch = m_model.outputSingle(it.next().getFeatureMatrix());
      int currentBatchSize = predBatch.shape()[0];

      // Build weka distribution output
      for (int i = 0; i < currentBatchSize; i++) {
        for (int j = 0; j < insts.numClasses(); j++) {
          preds[i + offset][j] = predBatch.getDouble(i, j);
        }
      }
      offset += currentBatchSize; // add batchsize as offset
      boolean hasInstancesLeft = offset < insts.numInstances();
      next = it.hasNext() || hasInstancesLeft;
    }



    // Fix classes
    for (int i = 0; i < preds.length; i++) {
      // only normalise if we're dealing with classification
      if (preds[i].length > 1) {
        weka.core.Utils.normalize(preds[i]);
      } else {
        // Rescale numeric classes with the computed coefficients in the initialization phase
        preds[i][0] = preds[i][0] * m_x1 + m_x0;
      }
    }
    return preds;
  }

  /**
   * Apply the filters to the given Instances
   * @param insts Instances that are going to be filtered
   * @return Filtered Instances
   * @throws Exception Filter could not be applied
   */
  private Instances applyFilters(Instances insts) throws Exception {
    // Filter the instance
    insts = Filter.useFilter(insts, m_replaceMissing);
    insts = Filter.useFilter(insts, m_nominalToBinary);
    if (m_Filter != null) {
      insts = Filter.useFilter(insts, m_Filter);
    }
    return insts;
  }

  /**
   * Get the {@link MultiLayerNetwork} model
   * @return MultiLayerNetwork instance
   */
  public ComputationGraph getModel() {
    return m_model;
  }

  /**
   * Returns a string describing the model.
   *
   * @return the model string
   */
  @Override
  public String toString() {

    if (m_replaceMissing != null) {
      return m_model.getConfiguration().toYaml();
    }
    return null;
  }
}
