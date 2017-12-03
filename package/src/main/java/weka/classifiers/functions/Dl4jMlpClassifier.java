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

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.output.CountingOutputStream;
import org.apache.commons.io.output.NullOutputStream;
import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseOutputLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.CachingDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.cache.InFileDataSetCache;
import org.nd4j.linalg.dataset.api.iterator.cache.InMemoryDataSetCache;
import org.nd4j.linalg.factory.Nd4j;
import weka.classifiers.IterativeClassifier;
import weka.classifiers.RandomizableClassifier;
import weka.classifiers.rules.ZeroR;
import weka.core.BatchPredictor;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.CapabilitiesHandler;
import weka.core.EmptyIteratorException;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.InvalidValidationPercentageException;
import weka.core.MissingOutputLayerException;
import weka.core.OptionMetadata;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.UnsupportedAttributeTypeException;
import weka.core.WekaException;
import weka.core.WekaPackageManager;
import weka.core.WrongIteratorException;
import weka.dl4j.CacheMode;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.earlystopping.EarlyStopping;
import weka.dl4j.iterators.instance.AbstractInstanceIterator;
import weka.dl4j.iterators.instance.Convolutional;
import weka.dl4j.iterators.instance.DefaultInstanceIterator;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.iterators.instance.ResizeImageInstanceIterator;
import weka.dl4j.layers.ConvolutionLayer;
import weka.dl4j.layers.OutputLayer;
import weka.dl4j.layers.SubsamplingLayer;
import weka.dl4j.listener.EpochListener;
import weka.dl4j.zoo.CustomNet;
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

/**
 * A wrapper for DeepLearning4j that can be used to train a multi-layer perceptron.
 *
 * @author Christopher Beckham
 * @author Eibe Frank
 * @author Steven Lang
 */
@Slf4j
public class Dl4jMlpClassifier extends RandomizableClassifier
    implements BatchPredictor, CapabilitiesHandler, IterativeClassifier {

  /** filter: Normalize training data */
  public static final int FILTER_NORMALIZE = 0;
  /** filter: Standardize training data */
  public static final int FILTER_STANDARDIZE = 1;
  /** filter: No normalization/standardization */
  public static final int FILTER_NONE = 2;
  /** The filter to apply to the training data */
  public static final Tag[] TAGS_FILTER = {
    new Tag(FILTER_NORMALIZE, "Normalize training data"),
    new Tag(FILTER_STANDARDIZE, "Standardize training data"),
    new Tag(FILTER_NONE, "No normalization/standardization"),
  };
  /** The ID used for serializing this class. */
  protected static final long serialVersionUID = -6363254116597574265L;
  /** Filter used to replace missing values. */
  protected ReplaceMissingValues replaceMissingFilter;
  /** Filter used to normalize or standardize the data. */
  protected Filter filter;
  /** Filter used to convert nominal attributes to binary numeric attributes. */
  protected NominalToBinary nominalToBinaryFilter;
  /** ZeroR classifier, just in case we don't actually have any data to train a network. */
  protected ZeroR zeroR;
  /** The actual neural network model. */
  protected transient ComputationGraph model;
  /** The model zoo model. */
  protected ZooModel zooModel = new CustomNet();
  /** The size of the serialized network model in bytes. */
  protected long modelSize;
  /** The file that log information will be written to. */
  protected File logFile = new File(
          Paths.get(WekaPackageManager.WEKA_HOME.getAbsolutePath(), "network.log").toString());
  /** The layers of the network. */
  protected Layer[] layers = new Layer[] {new OutputLayer()};
  /** The configuration of the network. */
  protected NeuralNetConfiguration netConfig = new NeuralNetConfiguration();
  /** The configuration for early stopping. */
  protected EarlyStopping earlyStopping = new EarlyStopping();
  /** The number of epochs to perform. */
  protected int numEpochs = 10;
  /** The number of epochs that have been performed. */
  protected int numEpochsPerformed;
  /** The dataset trainIterator. */
  protected transient DataSetIterator trainIterator;
  /** The training instances (set to null when done() is called). */
  protected Instances trainData;
  /** The instance iterator to use. */
  protected AbstractInstanceIterator instanceIterator = new DefaultInstanceIterator();
  /** Queue size for AsyncDataSetIterator (if < 1, AsyncDataSetIterator is not used) */
  protected int queueSize = 0;
  /** Whether to normalize/standardize/neither */
  protected int filterType = FILTER_STANDARDIZE;
  /** Coefficient x0 used for normalizing the class */
  protected double x0 = 0.0;
  /** Coefficient x1 used for normalizing the class */
  protected double x1 = 1.0;
  /** Caching mode to use for loading data */
  protected CacheMode cacheMode = CacheMode.NONE;
  /** Training listener list */
  private IterationListener iterationListener = new EpochListener();

  /** Default constructor fixing log file if WEKA_HOME variable is not set. */
  public Dl4jMlpClassifier() {
    Nd4j.getMemoryManager().setAutoGcWindow(10000);
  }

  /**
   * The main method for running this class.
   *
   * @param argv the command-line arguments
   */
  public static void main(String[] argv) {
    runClassifier(new Dl4jMlpClassifier(), argv);
  }

  /**
   * Split the dataset into p% train an (100-p)% test set
   *
   * @param data Input data
   * @param p train percentage
   * @return Array of instances: (0) Train, (1) Test
   * @throws Exception Filterapplication went wrong
   */
  public static Instances[] splitTrainVal(Instances data, double p) throws Exception {
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

    return new Instances[] {train, test};
  }

  public String globalInfo() {
    return "Classification and regression with multilayer perceptrons using DeepLearning4J.\n"
        + "Evaluations after each epoch are written to the log file.\n\n"
        + "Iterator usage\n"
        + "- DefaultInstanceIterator: Simple ARFF files without spatial interpretation\n"
        + "- ConvolutionalInstanceIterator: ARFF files with spatial interpretation\n"
        + "- ImageInstanceIterator: ARFF files containing meta-data linking to actual images\n"
        + "(See also https://deeplearning.cms.waikato.ac.nz/user-guide/data/ )";
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
      result.enable(Capability.STRING_ATTRIBUTES);
    } else {
      result.enable(Capability.NOMINAL_ATTRIBUTES);
      result.enable(Capability.NUMERIC_ATTRIBUTES);
      result.enable(Capability.DATE_ATTRIBUTES);
      result.enable(Capability.MISSING_VALUES);
      result.enableDependency(Capability.STRING_ATTRIBUTES); // User might switch to ImageDSI in GUI
    }

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.NUMERIC_CLASS);
    result.enable(Capability.DATE_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

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
    if (replaceMissingFilter != null) {
      ModelSerializer.writeModel(model, cos, false);
    }
    modelSize = cos.getByteCount();

    // default serialization
    oos.defaultWriteObject();

    // actually write the network
    if (replaceMissingFilter != null) {
      ModelSerializer.writeModel(model, oos, false);
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
      if (replaceMissingFilter != null) {
        File tmpFile = File.createTempFile("restore", "multiLayer");
        tmpFile.deleteOnExit();
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(tmpFile));
        long remaining = modelSize;
        while (remaining > 0) {
          int bsize = 10024;
          if (remaining < 10024) {
            bsize = (int) remaining;
          }
          byte[] buffer = new byte[bsize];
          int len = ois.read(buffer);
          if (len == -1) {
            throw new IOException(
                "Reached end of network model prematurely during deserialization.");
          }
          bos.write(buffer, 0, len);
          remaining -= len;
        }
        bos.flush();
        model = ModelSerializer.restoreComputationGraph(tmpFile, false);
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
    return logFile;
  }

  /**
   * Set the log file
   *
   * @param logFile the log file
   */
  @OptionMetadata(
    displayName = "log file",
    description =
        "The name of the log file to write loss information to "
            + "(default = $WEKA_HOME/network.log).",
    commandLineParamName = "logFile",
    commandLineParamSynopsis = "-logFile <string>",
    displayOrder = 1
  )
  public void setLogFile(File logFile) {
    this.logFile = logFile;
  }

  public Layer[] getLayers() {
    return layers;
  }

  @OptionMetadata(
    displayName = "layer specification.",
    description = "The specification of a layer. This option can be used multiple times.",
    commandLineParamName = "layer",
    commandLineParamSynopsis = "-layer <string>",
    displayOrder = 2
  )
  public void setLayers(Layer[] layers) {
    validateLayers(layers);

    // If something changed, set zoomodel to CustomNet
    if (!Arrays.deepEquals(layers, this.layers)) {
      setCustomNet();
    }
    fixDuplicateLayerNames(layers);
    this.layers = layers;
  }

  /**
   * Validate whether the layers comply with the currently chosen instance iterator
   *
   * @param layers New set of layers
   */
  protected void validateLayers(Layer[] layers) {
    // Check if the layers contain convolution/subsampling
    Set<Layer> layerSet = new HashSet<>(Arrays.asList(layers));
    final boolean containsConvLayer = layerSet.stream().allMatch(this::isNDLayer);

    final boolean isConvItertor = getInstanceIterator() instanceof Convolutional;
    if (containsConvLayer && !isConvItertor) {
      throw new RuntimeException(
          "A convolution/subsampling layer was set using "
              + "the wrong instance iterator. Please select either "
              + "ImageInstanceIterator for image files or "
              + "ConvolutionInstanceIterator for ARFF files.");
    }
  }

  /**
   * Check if a given layer is a convolutional/subsampling layer
   *
   * @param layer Layer to check
   * @return True if layer is convolutional/subsampling
   */
  private boolean isNDLayer(Layer layer) {
    return layer instanceof ConvolutionLayer
        || layer instanceof SubsamplingLayer
        || layer instanceof org.deeplearning4j.nn.conf.layers.ConvolutionLayer
        || layer instanceof org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
  }

  /**
   * Check if layer names are duplicate. If so, correct them by appending indices
   *
   * @param layers Array of network layer
   */
  protected void fixDuplicateLayerNames(Layer[] layers) {
    Set<String> names = Arrays.stream(layers).map(Layer::getLayerName).collect(Collectors.toSet());

    for (String name : names) {
      // Find duplicates with the same name
      List<Layer> duplicates =
          Arrays.stream(layers)
              .filter(l -> name.equals(l.getLayerName()))
              .collect(Collectors.toList());

      // If no duplicates were found, continue
      if (duplicates.size() == 1) {
        continue;
      }

      // For each duplicate add an index
      for (int i = 0; i < duplicates.size(); i++) {
        duplicates.get(i).setLayerName(name + " " + (i + 1));
      }
    }
  }

  public int getNumEpochs() {
    return numEpochs;
  }

  @OptionMetadata(
    description = "The number of epochs to perform.",
    displayName = "number of epochs",
    commandLineParamName = "numEpochs",
    commandLineParamSynopsis = "-numEpochs <int>",
    displayOrder = 4
  )
  public void setNumEpochs(int numEpochs) {
    this.numEpochs = numEpochs;
  }

  @OptionMetadata(
    description = "The instance trainIterator to use.",
    displayName = "instance iterator",
    commandLineParamName = "iterator",
    commandLineParamSynopsis = "-iterator <string>",
    displayOrder = 6
  )
  public AbstractInstanceIterator getInstanceIterator() {
    return instanceIterator;
  }

  public void setInstanceIterator(AbstractInstanceIterator iterator) {
    instanceIterator = iterator;
  }

  @OptionMetadata(
    description = "The neural network configuration to use.",
    displayName = "network configuration",
    commandLineParamName = "config",
    commandLineParamSynopsis = "-config <string>",
    displayOrder = 7
  )
  public NeuralNetConfiguration getNeuralNetConfiguration() {
    return netConfig;
  }

  public void setNeuralNetConfiguration(NeuralNetConfiguration config) {
    if (!config.equals(netConfig)) {
      setCustomNet();
    }
    netConfig = config;
  }

  /** Reset zoomodel to CustomNet */
  private void setCustomNet() {
    if (useZooModel()) {
      zooModel = new CustomNet();
    }
  }

  @OptionMetadata(
    description = "The early stopping configuration to use.",
    displayName = "early stopping configuration",
    commandLineParamName = "early-stopping",
    commandLineParamSynopsis = "-early-stopping <string>",
    displayOrder = 7
  )
  public EarlyStopping getEarlyStopping() {
    return earlyStopping;
  }

  public void setEarlyStopping(EarlyStopping config) {
    earlyStopping = config;
  }

  @OptionMetadata(
    description = "The type of normalization to perform.",
    displayName = "attribute normalization",
    commandLineParamName = "normalization",
    commandLineParamSynopsis = "-normalization <int>",
    displayOrder = 8
  )
  public SelectedTag getFilterType() {
    return new SelectedTag(filterType, TAGS_FILTER);
  }

  public void setFilterType(SelectedTag newType) {
    if (newType.getTags() == TAGS_FILTER) {
      filterType = newType.getSelectedTag().getID();
    }
  }

  public int getQueueSize() {
    return queueSize;
  }

  @OptionMetadata(
    description =
        "The queue size for asynchronous data transfer (default: 0, synchronous transfer).",
    displayName = "queue size for asynchronous data transfer",
    commandLineParamName = "queueSize",
    commandLineParamSynopsis = "-queueSize <int>",
    displayOrder = 9
  )
  public void setQueueSize(int QueueSize) {
    queueSize = QueueSize;
  }

  /**
   * The method used to train the classifier.
   *
   * @param data set of instances serving as training data
   * @throws Exception if something goes wrong in the training process
   */
  @Override
  public void buildClassifier(Instances data) throws Exception {
    log.info("Building on {} training instances", data.numInstances());

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
    try {
      getCapabilities().testWithFail(data);
    } catch (UnsupportedAttributeTypeException uate) {
      if (data.numAttributes() == 2 && data.attribute(0).isString()) {
        throw new UnsupportedAttributeTypeException(
            "It seems like you have chosen an ARFF file containing the "
                + "image paths without setting an ImageInstanceIterator");
      } else {
        throw uate;
      }
    }

    // Check basic network structure
    if (layers.length == 0) {
      throw new MissingOutputLayerException("No layers have been added!");
    }

    final Layer lastLayer = layers[layers.length - 1];
    if (!(lastLayer instanceof BaseOutputLayer)) {
      throw new MissingOutputLayerException("Last layer in network must be an output layer!");
    }

    // Apply preprocessing
    data = preProcessInput(data);

    // Split train/validation
    double valSplit = earlyStopping.getValidationSetPercentage();
    Instances trainData = null;
    Instances valData = null;
    if (useEarlyStopping()) {
      Instances[] insts = splitTrainVal(data, valSplit);
      trainData = insts[0];
      valData = insts[1];
      validateSplit(trainData, valData);
      DataSetIterator valIterator = getDataSetIterator(valData, CacheMode.NONE);
      earlyStopping.init(valIterator);

    } else {
      trainData = data;
    }

    if (trainData == null) {
      return;
    } else {
      this.trainData = trainData;
    }

    ClassLoader origLoader = Thread.currentThread().getContextClassLoader();
    try {
      Thread.currentThread().setContextClassLoader(this.getClass().getClassLoader());

      // If zoo model was set, use this model as internal MultiLayerNetwork
      if (useZooModel()) {
        createZooModel();
      } else {
        createModel();
      }
      // Setup the datasetiterators (needs to be done after the model initialization)
      trainIterator = getDataSetIterator(this.trainData);


      // Print model architecture
      if (getDebug()) {
        log.info(model.conf().toYaml());
      }

      // Set the iteration listener
      model.setListeners(getListener());

      numEpochsPerformed = 0;
    } finally {
      Thread.currentThread().setContextClassLoader(origLoader);
    }
  }

  private void validateSplit(Instances trainData, Instances valData) throws WekaException {
    if (earlyStopping.getValidationSetPercentage() < 10e-8) {
      // Use no validation set at all
      return;
    }
    int classIndex = trainData.classIndex();
    int valDataNumDinstinctClassValues = valData.numDistinctValues(classIndex);
    int trainDataNumDistinctClassValues = trainData.numDistinctValues(classIndex);
    if (trainData.numClasses() > 1
        && valDataNumDinstinctClassValues != trainDataNumDistinctClassValues) {
      throw new InvalidValidationPercentageException(
          "The validation data did not contain the same classes as the training data. "
              + "You should increase the validation percentage in the EarlyStopping configuration.");
    }
  }

  /**
   * Generates a DataSetIterator based on the given instances.
   *
   * @param data Input instances
   * @return DataSetIterator
   * @throws Exception
   */
  protected DataSetIterator getDataSetIterator(Instances data, CacheMode cm) throws Exception {
    DataSetIterator it = instanceIterator.getDataSetIterator(data, getSeed());

    // Use caching if set
    switch (cm) {
      case MEMORY: // Use memory as cache
        final InMemoryDataSetCache memCache = new InMemoryDataSetCache();
        it = new CachingDataSetIterator(it, memCache);
        break;
      case FILESYSTEM: // use filesystem as cache
        final String tmpDir = System.getProperty("java.io.tmpdir");
        final File cacheDir = Paths.get(tmpDir, "dataset-cache").toFile();
        cacheDir.delete(); // remove old existing cache
        final InFileDataSetCache fsCache = new InFileDataSetCache(cacheDir);
        it = new CachingDataSetIterator(it, fsCache);
        break;
    }

    // Use async dataset iteration if queue size was set
    if (queueSize > 0) {
      it = new AsyncDataSetIterator(it, queueSize);
      if (!it.hasNext()) {
        throw new RuntimeException("AsyncDataSetIterator could not load any datasets.");
      }
    }
    return it;
  }

  /**
   * Generates a DataSetIterator based on the given instances.
   *
   * @param data Input instances
   * @return DataSetIterator
   * @throws Exception
   */
  protected DataSetIterator getDataSetIterator(Instances data) throws Exception {
    return getDataSetIterator(data, cacheMode);
  }

  /**
   * Apply weka filter preprocessing to the input data.
   *
   * @param data Data as weka instances
   * @return Preprocessed instances
   * @throws Exception Preprocessing failed
   */
  private Instances preProcessInput(Instances data) throws Exception {
    // Remove instances with missing class and check that instances and
    // predictor attributes remain.
    data = new Instances(data);
    data.deleteWithMissingClass();
    zeroR = null;
    int numSamples = data.numInstances();
    if (numSamples == 0 || data.numAttributes() < 2) {
      zeroR = new ZeroR();
      zeroR.buildClassifier(data);
      return null;
    }

    // Retrieve two different class values used to determine filter
    // transformation
    double y0 = data.instance(0).classValue();
    int index = 1;
    while (index < numSamples && data.instance(index).classValue() == y0) {
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
    x1 = (y0 - y1) / (z0 - z1); // no division by zero, since y0 != y1
    // guaranteed => z0 != z1 ???
    x0 = (y0 - x1 * z0); // = y1 - x1 * z1

    // Randomize the data, just in case
    Random rand = new Random(getSeed());
    data.randomize(rand);
    return data;
  }

  /**
   * Initialize {@link ReplaceMissingValues}, {@link NominalToBinary} and {@link Standardize} or
   * {@link Normalize} filters
   *
   * @param data Input data to set the input formal of the filters
   * @return Transformed data
   * @throws Exception Filter can not be initialized
   */
  private Instances initFilters(Instances data) throws Exception {
    // Replace missing values
    replaceMissingFilter = new ReplaceMissingValues();
    replaceMissingFilter.setInputFormat(data);
    data = Filter.useFilter(data, replaceMissingFilter);

    // Replace nominal attributes by binary numeric attributes.
    nominalToBinaryFilter = new NominalToBinary();
    nominalToBinaryFilter.setInputFormat(data);
    data = Filter.useFilter(data, nominalToBinaryFilter);

    // Standardize or normalize (as requested), including the class

    if (filterType == FILTER_STANDARDIZE) {
      filter = new Standardize();
      filter.setOptions(new String[] {"-unset-class-temporarily"});
      filter.setInputFormat(data);
      data = Filter.useFilter(data, filter);
    } else if (filterType == FILTER_NORMALIZE) {
      filter = new Normalize();
      filter.setOptions(new String[] {"-unset-class-temporarily"});
      filter.setInputFormat(data);
      data = Filter.useFilter(data, filter);
    } else {
      filter = null;
    }

    return data;
  }

  /**
   * Build the Zoomodel instance
   *
   * @return ComputationGraph instance
   * @throws WekaException Either the .init operation on the current zooModel was not supported or
   *     the data shape does not fit the chosen zooModel
   */
  private void createZooModel() throws WekaException {
    final AbstractInstanceIterator it = getInstanceIterator();
    final boolean isImageIterator = it instanceof ImageInstanceIterator;

    // Make sure data is convolutional
    if (!isImageIterator) {
      throw new WrongIteratorException(
          "ZooModels currently only support images. " + "Please setup an ImageInstanceIterator.");
    }

    // Get the new width/heigth/channels from the iterator
    ImageInstanceIterator iii = (ImageInstanceIterator) it;
    int newWidth = iii.getWidth();
    int newHeight = iii.getHeight();
    int channels = iii.getNumChannels();
    boolean initSuccessful = false;
    while (!initSuccessful) {
      // Increase width and height
      int[] newShape = new int[] {channels, newHeight, newWidth};
      int[][] shapeWrap = new int[][] {newShape};
      setInstanceIterator(new ResizeImageInstanceIterator(iii, newWidth, newHeight));
      initSuccessful = initZooModel(trainData.numClasses(), getSeed(), shapeWrap);

      newWidth *= 1.2;
      newHeight *= 1.2;
      if (!initSuccessful) {
        log.warn(
            "The shape of the data did not fit the chosen "
                + "model. It was therefore resized to ({}x{}x{}).",
            channels,
            newHeight,
            newWidth);
      }
    }
  }

  private boolean initZooModel(int numClasses, long seed, int[][] newShape) {
    try {
      model = zooModel.init(numClasses, seed, newShape);
      return true;
    } catch (UnsupportedOperationException e) {
      throw new UnsupportedOperationException(
          "ZooModel was not set (CustomNet), but createZooModel could be called. Invalid situation",
          e);
    } catch (DL4JInvalidConfigException | DL4JInvalidInputException e) {
      return false;
    }
  }

  /**
   * Build the multilayer network defined by the networkconfiguration and the list of layers.
   *
   * @throws Exception
   */
  protected void createModel() throws Exception {
    final INDArray features = getFirstBatchFeatures(trainData);
    ComputationGraphConfiguration.GraphBuilder gb =
        netConfig
            .builder()
            .seed(getSeed())
            .inferenceWorkspaceMode(WorkspaceMode.SEPARATE)
            .trainingWorkspaceMode(WorkspaceMode.SEPARATE)
            .graphBuilder();

    // Set ouput size
    final Layer lastLayer = layers[layers.length - 1];
    final int nOut = trainData.numClasses();
    if (lastLayer instanceof BaseOutputLayer) {
      ((BaseOutputLayer) lastLayer).setNOut(nOut);
    }

    String currentInput = "input";
    gb.addInputs(currentInput);
    // Collect layers
    for (Layer layer : layers) {
      String lName = layer.getLayerName();
      gb.addLayer(lName, layer.clone(), currentInput);
      currentInput = lName;
    }
    gb.setOutputs(currentInput);
    gb.setInputTypes(InputType.inferInputType(features));

    ComputationGraphConfiguration conf = gb.pretrain(false).backprop(true).build();
    ComputationGraph model = new ComputationGraph(conf);
    model.init();
    this.model = model;
  }

  /**
   * Get a peak at the features of the {@code iterator}'s first batch using the given instances.
   *
   * @return Features of the first batch
   * @throws Exception
   */
  protected INDArray getFirstBatchFeatures(Instances data) throws Exception {
    final DataSetIterator it = getDataSetIterator(data, CacheMode.NONE);
    if (!it.hasNext()) {
      throw new RuntimeException("Iterator was unexpectedly empty.");
    }
    final INDArray features = it.next().getFeatures();
    it.reset();
    return features;
  }

  /**
   * Get the iterationlistener
   *
   * @throws Exception
   */
  protected List<IterationListener> getListener() throws Exception {
    int numSamples = trainData.numInstances();
    List<IterationListener> listeners = new ArrayList<>();

    // Initialize weka listener
    if (iterationListener instanceof weka.dl4j.listener.EpochListener) {
      int numEpochs = getNumEpochs();
      ((EpochListener) iterationListener)
          .init(trainData.numClasses(), numEpochs, numSamples, trainIterator, earlyStopping.getValDataSetIterator());
      ((EpochListener) iterationListener).setLogFile(logFile);
      listeners.add(iterationListener);
    } else {
      listeners.add(iterationListener);
    }
    return listeners;
  }

  /** Perform another epoch. */
  public boolean next() throws Exception {

    if (numEpochsPerformed >= getNumEpochs() || zeroR != null || trainData == null) {
      return false;
    }

    // Check if trainIterator was reset properly
    if (!trainIterator.hasNext()) {
      throw new EmptyIteratorException(
          "The iterator has no next elements " + "at the beginning of the epoch.");
    }

    ClassLoader origLoader = Thread.currentThread().getContextClassLoader();
    try {
      Thread.currentThread().setContextClassLoader(this.getClass().getClassLoader());
      StopWatch sw = new StopWatch();
      sw.start();
      model.fit(trainIterator);
      trainIterator.reset();
      sw.stop();
      log.info("Epoch {}/{} took {}", numEpochsPerformed, numEpochs, sw.toString());
      numEpochsPerformed++;
    } finally {
      Thread.currentThread().setContextClassLoader(origLoader);
    }

    // Evaluate early stopping
    if (useEarlyStopping()) {
      boolean continueTraining = earlyStopping.evaluate(model);
      if (!continueTraining) {
        log.info(
            "Early stopping has stopped the training process. The "
                + "validation has not improved anymore after {} epochs. Training "
                + "finished.",
            earlyStopping.getMaxEpochsNoImprovement());
      }
      return continueTraining;
    }

    return true;
  }

  /**
   * Use early stopping only if valid split percentage
   *
   * @return True if early stopping should be used
   */
  public boolean useEarlyStopping() {
    double p = earlyStopping.getValidationSetPercentage();
    return 0 < p && p < 100;
  }

  /** Clean up after learning. */
  public void done() {

    trainData = null;
  }

  /**
   * Get the modelzoo model
   *
   * @return The modelzoo model object
   */
  public ZooModel getZooModel() {
    return zooModel;
  }

  /**
   * Set the modelzoo zooModel
   *
   * @param zooModel The predefined zooModel
   */
  @OptionMetadata(
    displayName = "zooModel",
    description = "The model-architecture to choose from the modelzoo " + "(default = no model).",
    commandLineParamName = "zooModel",
    commandLineParamSynopsis = "-zooModel <string>",
    displayOrder = 11
  )
  public void setZooModel(ZooModel zooModel) {
    if (zooModel instanceof GoogLeNet || zooModel instanceof FaceNetNN4Small2) {
      throw new RuntimeException(
          "The zoomodel you have selected is currently"
              + " not supported! Please select another one.");
    }

    this.zooModel = zooModel;

    try {
      // Try to parse the layers so the user can change them afterwards
      final int dummyNumLabels = 2;
      ComputationGraph tmpCg = zooModel.init(dummyNumLabels, getSeed(), zooModel.getShape());
      tmpCg.init();
      layers =
          Arrays.stream(tmpCg.getLayers())
              .map(l -> l.conf().getLayer())
              .collect(Collectors.toList())
              .toArray(new Layer[tmpCg.getLayers().length]);
    } catch (Exception e) {
      if (!(zooModel instanceof CustomNet)) {
        log.error("Could not set layers from zoomodel.", e);
      }
    }
  }

  public IterationListener getIterationListener() {
    return iterationListener;
  }

  @OptionMetadata(
    displayName = "set the iteration listener",
    description = "Set the iteration listener.",
    commandLineParamName = "iteration-listener",
    commandLineParamSynopsis = "-iteration-listener <string>",
    displayOrder = 12
  )
  public void setIterationListener(IterationListener l) {
    iterationListener = l;
  }

  public CacheMode getCacheMode() {
    return cacheMode;
  }

  @OptionMetadata(
    displayName = "set the cache mode",
    description = "Set the cache mode.",
    commandLineParamName = "cache-mode",
    commandLineParamSynopsis = "-cache-mode <string>",
    displayOrder = 13
  )
  public void setCacheMode(CacheMode cm) {
    cacheMode = cm;
  }

  /**
   * Performs efficient batch prediction
   *
   * @return true
   */
  @Override
  public boolean implementsMoreEfficientBatchPrediction() {
    return true;
  }

  /**
   * The method to use when making a prediction for a test instance. Use distributionsForInstances()
   * instead for speed if possible.
   *
   * @param inst the instance to get a prediction for
   * @return the class probability estimates (if the class is nominal) or the numeric prediction (if
   *     it is numeric)
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
   * @return the class probability estimates (if the class is nominal) or the numeric predictions
   *     (if it is numeric)
   * @throws Exception if something goes wrong at prediction time
   */
  @Override
  public double[][] distributionsForInstances(Instances insts) throws Exception {

    // Do we only have a ZeroR model?
    if (zeroR != null) {
      return zeroR.distributionsForInstances(insts);
    }

    // Process input data to have the same filters applied as the training data
    insts = applyFilters(insts);

    // Get predictions
    final DataSetIterator it = getDataSetIterator(insts, CacheMode.NONE);
    double[][] preds = new double[insts.numInstances()][insts.numClasses()];

    int offset = 0;
    boolean next = it.hasNext();

    // Get predictions batch-wise
    while (next) {
      INDArray predBatch = model.outputSingle(it.next().getFeatureMatrix());
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
        preds[i][0] = preds[i][0] * x1 + x0;
      }
    }
    return preds;
  }

  /**
   * Apply the filters to the given Instances
   *
   * @param insts Instances that are going to be filtered
   * @return Filtered Instances
   * @throws Exception Filter could not be applied
   */
  protected Instances applyFilters(Instances insts) throws Exception {
    // Filter the instance
    insts = Filter.useFilter(insts, replaceMissingFilter);
    insts = Filter.useFilter(insts, nominalToBinaryFilter);
    if (filter != null) {
      insts = Filter.useFilter(insts, filter);
    }
    return insts;
  }

  /**
   * Get the {@link MultiLayerNetwork} model
   *
   * @return MultiLayerNetwork instance
   */
  public ComputationGraph getModel() {
    return model;
  }

  /**
   * Returns a string describing the model.
   *
   * @return the model string
   */
  @Override
  public String toString() {

    if (replaceMissingFilter != null) {
      return model.getConfiguration().toYaml();
    }
    return null;
  }

  /**
   * Check if the user has selected to use a zoomodel
   *
   * @return True if zoomodel is not CustomNet
   */
  private boolean useZooModel() {
    return !(zooModel instanceof CustomNet);
  }
}
