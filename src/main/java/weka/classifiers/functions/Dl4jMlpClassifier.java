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
import java.net.URI;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.io.output.CountingOutputStream;
import org.apache.commons.io.output.NullOutputStream;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.LoggerContext;
import org.apache.logging.log4j.core.config.ConfigurationSource;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.BaseOutputLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.CachingDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.cache.InFileDataSetCache;
import org.nd4j.linalg.dataset.api.iterator.cache.InMemoryDataSetCache;
import org.nd4j.linalg.factory.Nd4j;
import weka.classifiers.IterativeClassifier;
import weka.classifiers.RandomizableClassifier;
import weka.classifiers.functions.dl4j.Utils;
import weka.classifiers.rules.ZeroR;
import weka.core.BatchPredictor;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.CapabilitiesHandler;
import weka.core.EmptyIteratorException;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.InvalidLayerConfigurationException;
import weka.core.InvalidNetworkArchitectureException;
import weka.core.InvalidValidationPercentageException;
import weka.core.MissingOutputLayerException;
import weka.core.OptionMetadata;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.WekaException;
import weka.core.WekaPackageManager;
import weka.core.WrongIteratorException;
import weka.dl4j.CacheMode;
import weka.dl4j.ConvolutionMode;
import weka.dl4j.NeuralNetConfiguration;
import weka.dl4j.earlystopping.EarlyStopping;
import weka.dl4j.iterators.instance.AbstractInstanceIterator;
import weka.dl4j.iterators.instance.DefaultInstanceIterator;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.iterators.instance.ResizeImageInstanceIterator;
import weka.dl4j.iterators.instance.api.ConvolutionalIterator;
import weka.dl4j.iterators.instance.sequence.text.cnn.CnnTextEmbeddingInstanceIterator;
import weka.dl4j.layers.ConvolutionLayer;
import weka.dl4j.layers.FeedForwardLayer;
import weka.dl4j.layers.GlobalPoolingLayer;
import weka.dl4j.layers.Layer;
import weka.dl4j.layers.OutputLayer;
import weka.dl4j.layers.SubsamplingLayer;
import weka.dl4j.listener.EpochListener;
import weka.dl4j.listener.TrainingListener;
import weka.dl4j.zoo.CustomNet;
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
@Log4j2
public class Dl4jMlpClassifier extends RandomizableClassifier
    implements BatchPredictor, CapabilitiesHandler, IterativeClassifier {

  /**
   * The ID used for serializing this class.
   */
  private static final long serialVersionUID = -6363254116597574265L;
  /**
   * filter: Normalize training data
   */
  public static final int FILTER_NORMALIZE = 0;
  /**
   * filter: Standardize training data
   */
  public static final int FILTER_STANDARDIZE = 1;
  /**
   * filter: No normalization/standardization
   */
  public static final int FILTER_NONE = 2;
  /**
   * The filter to apply to the training data
   */
  public static final Tag[] TAGS_FILTER = {
      new Tag(FILTER_NORMALIZE, "Normalize training data"),
      new Tag(FILTER_STANDARDIZE, "Standardize training data"),
      new Tag(FILTER_NONE, "No normalization/standardization"),
  };
  /**
   * Filter used to replace missing values.
   */
  protected ReplaceMissingValues replaceMissingFilter;
  /**
   * Filter used to normalize or standardize the data.
   */
  protected Filter filter;
  /**
   * Filter used to convert nominal attributes to binary numeric attributes.
   */
  protected NominalToBinary nominalToBinaryFilter;
  /**
   * ZeroR classifier, just in case we don't actually have any data to train a network.
   */
  protected ZeroR zeroR;
  /**
   * The actual neural network model.
   */
  protected transient ComputationGraph model;
  /**
   * The model zoo model.
   */
  protected ZooModel zooModel = new CustomNet();
  /**
   * The size of the serialized network model in bytes.
   */
  protected long modelSize;
  /**
   * The file that log information will be written to.
   */
  protected File logFile =
      new File(Paths.get(WekaPackageManager.WEKA_HOME.getAbsolutePath(), "network.log").toString());
  /**
   * The layers of the network.
   */
  protected transient Layer[] layers = new Layer[]{new OutputLayer()};
  /**
   * The configuration of the network.
   */
  protected NeuralNetConfiguration netConfig = new NeuralNetConfiguration();
  /**
   * The configuration for early stopping.
   */
  protected EarlyStopping earlyStopping = new EarlyStopping();
  /**
   * The number of epochs to perform.
   */
  protected int numEpochs = 10;
  /**
   * The number of epochs that have been performed.
   */
  protected int numEpochsPerformed;
  /**
   * The dataset trainIterator.
   */
  protected transient DataSetIterator trainIterator;
  /**
   * The training instances (set to null when done() is called).
   */
  protected Instances trainData;
  /**
   * The instance iterator to use.
   */
  protected AbstractInstanceIterator instanceIterator = new DefaultInstanceIterator();
  /**
   * Queue size for AsyncDataSetIterator (if < 1, AsyncDataSetIterator is not used)
   */
  protected int queueSize = 0;
  /**
   * Whether to normalize/standardize/neither
   */
  protected int filterType = FILTER_STANDARDIZE;
  /**
   * Coefficient x0 used for normalizing the class
   */
  protected double x0 = 0.0;
  /**
   * Coefficient x1 used for normalizing the class
   */
  protected double x1 = 1.0;
  /**
   * Caching mode to use for loading data
   */
  protected CacheMode cacheMode = CacheMode.MEMORY;
  /**
   * Training listener list
   */
  protected TrainingListener iterationListener = new EpochListener();

  /**
   * Flag indicating if initialization is finished.
   */
  protected boolean isInitializationFinished = false;

  /**
   * List of indices to store the label order.
   */
  protected int[] labelSortIndex;

  /**
   * Default constructor
   */
  public Dl4jMlpClassifier() {
    loadLoggerIfConfigMissing();
  }

  /**
   * Load the log4j2.xml specified in the package sources if no configuration is currently set.
   */
  private static void loadLoggerIfConfigMissing() {
    LoggerContext context = (LoggerContext) LogManager
        .getContext(false);
    ConfigurationSource configuration = context.getConfiguration().getConfigurationSource();
    if (ConfigurationSource.NULL_SOURCE.equals(configuration)) {
      // Use log4j2.xml shipped with the package ...
      String wekaHomeDir = WekaPackageManager.getPackageHome().getPath();
      URI uri = Paths.get(wekaHomeDir, "wekaDeeplearning4j", "src", "main", "resources",
          "log4j2.xml").toUri();
      context.setConfigLocation(uri);
      log.info("Logging configuration loaded from source: {}", uri.toString());
    }
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
    // Randomize data
    Randomize rand = new Randomize();
    rand.setInputFormat(data);
    rand.setRandomSeed(42);
    data = Filter.useFilter(data, rand);

    // Remove testpercentage from data to get the train set
    RemovePercentage rp = new RemovePercentage();
    rp.setInputFormat(data);
    rp.setPercentage(p);
    Instances train = Filter.useFilter(data, rp);

    // Remove trainpercentage from data to get the test set
    rp = new RemovePercentage();
    rp.setInputFormat(data);
    rp.setPercentage(p);
    rp.setInvertSelection(true);
    Instances test = Filter.useFilter(data, rp);

    return new Instances[]{train, test};
  }

  public String globalInfo() {
    return "Classification and regression with multilayer perceptrons using DeepLearning4J.\n"
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
    if (getInstanceIterator() instanceof ImageInstanceIterator
        || getInstanceIterator() instanceof CnnTextEmbeddingInstanceIterator) {
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
   */
  private void writeObject(ObjectOutputStream oos) throws IOException {
    // figure out size of the written network
    CountingOutputStream cos = new CountingOutputStream(new NullOutputStream());
    if (isInitializationFinished) {
      ModelSerializer.writeModel(model, cos, false);
    }
    modelSize = cos.getByteCount();

    // default serialization
    oos.defaultWriteObject();

    // Write layer configurations
    String[] layerConfigs = new String[layers.length];
    for (int i = 0; i < layers.length; i++) {
      layerConfigs[i] = layers[i].getClass().getName() + "::" + weka.core.Utils
          .joinOptions(layers[i].getOptions());
    }
    oos.writeObject(layerConfigs);

    // actually write the network
    if (isInitializationFinished) {
      ModelSerializer.writeModel(model, oos, false);
    }
  }

  /**
   * Custom deserialization method
   *
   * @param ois the object input stream
   */
  private void readObject(ObjectInputStream ois) throws ClassNotFoundException, IOException {
    ClassLoader origLoader = Thread.currentThread().getContextClassLoader();
    try {
      Thread.currentThread().setContextClassLoader(this.getClass().getClassLoader());
      // default deserialization
      ois.defaultReadObject();

      // Restore the layers
      String[] layerConfigs = (String[]) ois.readObject();
      layers = new Layer[layerConfigs.length];
      for (int i = 0; i < layerConfigs.length; i++) {
        String layerConfigString = layerConfigs[i];
        String[] split = layerConfigString.split("::");
        String clsName = split[0];
        String layerConfig = split[1];
        String[] options = weka.core.Utils.splitOptions(layerConfig);
        layers[i] = (Layer) weka.core.Utils.forName(Layer.class, clsName, options);
      }

      // restore the network model
      if (isInitializationFinished) {
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
    } catch (Exception e) {
      log.error("Failed to restore serialized model. Error: " + e.getMessage());
      e.printStackTrace();
    } finally {
      Thread.currentThread().setContextClassLoader(origLoader);
    }
  }

  /**
   * Generate the, for this model type, typical output layer.
   *
   * @return New OutputLayer object
   */
  protected Layer<? extends BaseOutputLayer> createOutputLayer() {
    return new OutputLayer();
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
  public void setLayers(Layer... layers) {
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
  protected void validateLayers(Layer[] layers) throws InvalidNetworkArchitectureException {
    // Check if the layers contain convolution/subsampling
    Set<Layer> layerSet = new HashSet<>(Arrays.asList(layers));
    final boolean containsConvLayer = layerSet.stream().allMatch(this::isNDLayer);

    final boolean isConvItertor = getInstanceIterator() instanceof ConvolutionalIterator;
    if (containsConvLayer && !isConvItertor) {
      throw new InvalidNetworkArchitectureException(
          "A convolution/subsampling layer was set using "
              + "the wrong instance iterator. Please select either "
              + "ImageInstanceIterator for image files or "
              + "ConvolutionInstanceIterator for ARFF files.");
    }

    // Check if conv layers have ConvolutionMode.Same for CnnTextEmbeddingInstanceIterator
    if (getInstanceIterator() instanceof CnnTextEmbeddingInstanceIterator) {
      for (Layer l : layerSet) {
        if (l instanceof ConvolutionLayer) {
          final ConvolutionLayer conv = (ConvolutionLayer) l;
          boolean correctMode = conv.getConvolutionMode().equals(ConvolutionMode.Same);
          if (!correctMode) {
            throw new RuntimeException(
                "CnnText iterators require ConvolutionMode.Same for all ConvolutionLayer. Layer "
                    + conv.getLayerName()
                    + " has ConvolutionMode: "
                    + conv.getConvolutionMode());
          }
        }
      }

      // Check that layers start with convolution
      if (layers.length > 0 && !(layers[0] instanceof ConvolutionLayer)) {
        throw new InvalidNetworkArchitectureException(
            "CnnText iterator requires ConvolutionLayer.");
      }
    }
  }

  /**
   * Check if a given layer is a convolutional/subsampling layer
   *
   * @param layer Layer to check
   * @return True if layer is convolutional/subsampling
   */
  protected boolean isNDLayer(Layer layer) {
    return layer instanceof ConvolutionLayer || layer instanceof SubsamplingLayer;
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

  /**
   * Reset zoomodel to CustomNet
   */
  protected void setCustomNet() {
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
      displayOrder = 12
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
      displayOrder = 30
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

    if (getDebug()) {
      log.info("Classifier: \n{}", toString());
    }

    boolean isContinue = true;
    while (isContinue) {
      // Next epoch
      isContinue = next();
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

    // If only class is present, build zeroR
    if (data.numAttributes() == 1 && data.classIndex() == 0) {
      zeroR = new ZeroR();
      zeroR.buildClassifier(data);
      return;
    }

    // Can classifier handle the data?
    getCapabilities().testWithFail(data);

    // Check basic network structure
    if (layers.length == 0) {
      throw new MissingOutputLayerException("No layers have been added!");
    }

    final Layer lastLayer = layers[layers.length - 1];
    org.deeplearning4j.nn.conf.layers.Layer lastLayerBackend = lastLayer.getBackend();
    if (!(lastLayerBackend instanceof BaseOutputLayer
        || lastLayerBackend instanceof LossLayer
        || lastLayerBackend instanceof ActivationLayer)) {
      throw new MissingOutputLayerException(
          "Last layer in network must be an output layer but was: " + lastLayerBackend
              .getClass().getSimpleName());
    }

    // Check if layers are valid
    validateLayers(layers);

    // Apply preprocessing
    data = preProcessInput(data);
    data = initEarlyStopping(data);
    saveLabelSortIndex(data);

    if (data != null) {
      trainData = data;
    } else {
      return;
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
      // Initialize iterator
      instanceIterator.initialize();

      // Setup the datasetiterators (needs to be done after the model initialization)
      trainIterator = getDataSetIterator(this.trainData);

      // Print model architecture
      if (getDebug()) {
        log.info(model.conf().toYaml());
      }

      // Set the iteration listener
      model.setListeners(getListener());

      numEpochsPerformed = 0;

      isInitializationFinished = true;
    } finally {
      Thread.currentThread().setContextClassLoader(origLoader);
    }
  }

  /**
   * Store the label sort index for mapping weka-labels to resorted dl4j-labels.
   *
   * @param data Input data
   */
  protected void saveLabelSortIndex(Instances data) {
    if (data.classAttribute().isNominal()) {
      // Save Label order as DL4J automatically sorts them
      List<String> labels = new ArrayList<>();
      int numClassValues = data.classAttribute().numValues();
      for (int i = 0; i < numClassValues; i++) {
        labels.add(data.classAttribute().value(i));
      }
      List<String> labelsSorted = new ArrayList<>(labels);
      Collections.sort(labelsSorted);
      labelSortIndex = new int[numClassValues];

      for (int i = 0; i < numClassValues; i++) {
        String label = labels.get(i);
        int sortedIndex = labelsSorted.indexOf(label);
        labelSortIndex[i] = sortedIndex;
      }
    }
  }

  /**
   * Initialize early stopping with the given data
   *
   * @param data Data
   * @return Augmented data - if early stopping applies, return train set without validation set
   */
  protected Instances initEarlyStopping(Instances data) throws Exception {
    // Split train/validation
    double valSplit = earlyStopping.getValidationSetPercentage();
    Instances trainData;
    Instances valData;
    if (useEarlyStopping()) {
      // Split in train and validation
      Instances[] insts = splitTrainVal(data, valSplit);
      trainData = insts[0];
      valData = insts[1];
      validateSplit(trainData, valData);
      DataSetIterator valIterator = getDataSetIterator(valData, cacheMode, "val");
      earlyStopping.init(valIterator);
    } else {
      // Keep the full data
      trainData = data;
    }

    return trainData;
  }

  /**
   * Validate a data split of train and validation data
   *
   * @param trainData Training data
   * @param valData Validation data
   * @throws WekaException Invalid validation split
   */
  protected void validateSplit(Instances trainData, Instances valData) throws WekaException {
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
   * @param cm Cache mode for the datasets
   * @param cacheDirSuffix suffix for the cache directory
   * @return DataSetIterator Iterator over dataset objects
   */
  protected DataSetIterator getDataSetIterator(Instances data, CacheMode cm, String cacheDirSuffix)
      throws Exception {
    DataSetIterator it = instanceIterator.getDataSetIterator(data, getSeed());

    // Use caching if set
    switch (cm) {
      case MEMORY: // Use memory as cache
        final InMemoryDataSetCache memCache = new InMemoryDataSetCache();
        it = new CachingDataSetIterator(it, memCache);
        break;
      case FILESYSTEM: // use filesystem as cache
        final String tmpDir = System.getProperty("java.io.tmpdir");
        final String suffix = cacheDirSuffix.isEmpty() ? "" : "-" + cacheDirSuffix;
        final File cacheDir = Paths.get(tmpDir, "dataset-cache" + suffix).toFile();
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
   * @param cm Cache mode for the datasets
   * @return DataSetIterator Iterator over dataset objects
   */
  protected DataSetIterator getDataSetIterator(Instances data, CacheMode cm) throws Exception {
    return getDataSetIterator(data, cm, "");
  }

  /**
   * Generates a DataSetIterator based on the given instances.
   *
   * @param data Input instances
   * @return DataSetIterator
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
  protected Instances preProcessInput(Instances data) throws Exception {
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
  protected Instances initFilters(Instances data) throws Exception {
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
      filter.setOptions(new String[]{"-unset-class-temporarily"});
      filter.setInputFormat(data);
      data = Filter.useFilter(data, filter);
    } else if (filterType == FILTER_NORMALIZE) {
      filter = new Normalize();
      filter.setOptions(new String[]{"-unset-class-temporarily"});
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
   * the data shape does not fit the chosen zooModel
   */
  protected void createZooModel() throws Exception {
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
      int[] newShape = new int[]{channels, newHeight, newWidth};
      setInstanceIterator(new ResizeImageInstanceIterator(iii, newWidth, newHeight));
      initSuccessful = initZooModel(trainData.numClasses(), getSeed(), newShape);

      newWidth *= 1.2;
      newHeight *= 1.2;
      if (!initSuccessful) {
        log.warn(
            "The data's shape did not fit the chosen "
                + "model's input. It was therefore resized to ({}x{}x{}).",
            channels,
            newHeight,
            newWidth);
      }
    }
  }

  protected boolean initZooModel(int numClasses, long seed, int[] newShape) throws Exception {
    try {
      ComputationGraph tmpModel = zooModel.init(numClasses, seed, newShape);
      // Make a dummy feed forward pass to check if the model dimensions fit at each layer
      Instances dummyData = new Instances(trainData);
      for (int i = 0; i < instanceIterator.getTrainBatchSize(); i++) {
        dummyData.add(trainData.get(i));
      }
      tmpModel.init();
      DataSetIterator iter = getDataSetIterator(dummyData);
      tmpModel.feedForward(iter.next().getFeatures(), false);

      // No Exception thrown -> set model to this zoo model and return true
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
   */
  protected void createModel() throws Exception {
    final INDArray features = getFirstBatchFeatures(trainData);
    ComputationGraphConfiguration.GraphBuilder gb =
        netConfig.builder().seed(getSeed()).graphBuilder();

    // Set ouput size
    final Layer lastLayer = layers[layers.length - 1];
    final int nOut = trainData.numClasses();
    if (lastLayer instanceof FeedForwardLayer) {
      ((FeedForwardLayer) lastLayer).setNOut(nOut);
    }

    if (getInstanceIterator() instanceof CnnTextEmbeddingInstanceIterator) {
      makeCnnTextLayerSetup(gb);
    } else {
      makeDefaultLayerSetup(gb);
    }

    gb.setInputTypes(InputType.inferInputType(features));
    ComputationGraphConfiguration conf = gb.pretrain(false).backprop(true).build();
    ComputationGraph model = new ComputationGraph(conf);
    model.init();
    this.model = model;
  }

  /**
   * Default layer setup: Create sequential layer network defined by the order of the layer list
   *
   * @param gb GraphBuilder object
   */
  protected void makeDefaultLayerSetup(GraphBuilder gb) {
    String currentInput = "input";
    gb.addInputs(currentInput);
    // Collect layers
    for (Layer layer : layers) {
      String lName = layer.getLayerName();
      gb.addLayer(lName, layer.getBackend().clone(), currentInput);
      currentInput = lName;
    }
    gb.setOutputs(currentInput);
  }

  /**
   * CnnText layer setup: Collect CNN layers and merge them in a {@link MergeVertex}.
   *
   * @param gb GraphBuilder object
   */
  protected void makeCnnTextLayerSetup(GraphBuilder gb)
      throws InvalidNetworkArchitectureException, InvalidLayerConfigurationException {
    String currentInput = "input";
    gb.addInputs(currentInput);

    // Collect all convolution layers defined until the first non conv layer
    List<ConvolutionLayer> convLayers = new ArrayList<>();
    int idx = 0;
    for (Layer l : layers) {
      if (l instanceof ConvolutionLayer) {
        final ConvolutionLayer convLayer = (ConvolutionLayer) l;
        validateCnnLayer(convLayer);
        convLayers.add(convLayer);
        gb.addLayer(convLayer.getLayerName(), convLayer.getBackend().clone(), currentInput);
        idx++;
      } else {
        break;
      }
    }

    // Check if next layer is GlobalPooling
    if (idx < layers.length && !(layers[idx] instanceof GlobalPoolingLayer)) {
      throw new InvalidNetworkArchitectureException(
          "For a CNN text setup, the list of convolution"
              + " layers must be followed by a GlobalPoolingLayer.");
    }

    // Collect names
    final String[] names =
        convLayers.stream().map(ConvolutionLayer::getLayerName).toArray(String[]::new);

    // Add merge vertex
    if (names.length > 0) {
      final String mergeVertexName = "merge";
      gb.addVertex(mergeVertexName, new MergeVertex(), names);
      currentInput = mergeVertexName;
    }

    // Collect layers
    for (
      /*use idx from above*/ ; idx < layers.length; idx++) {
      String lName = layers[idx].getLayerName();
      gb.addLayer(lName, layers[idx].getBackend().clone(), currentInput);
      currentInput = lName;
    }
    gb.setOutputs(currentInput);
  }

  /**
   * Validate CNN layers when using {@link CnnTextEmbeddingInstanceIterator}.
   *
   * @param cl Convolution Layer
   * @throws InvalidLayerConfigurationException Invalid configuration
   */
  protected void validateCnnLayer(ConvolutionLayer cl) throws InvalidLayerConfigurationException {
    final AbstractInstanceIterator iter = getInstanceIterator();
    if (iter instanceof CnnTextEmbeddingInstanceIterator) {
      CnnTextEmbeddingInstanceIterator cnnIter = (CnnTextEmbeddingInstanceIterator) iter;
      final int vectorSize =
          cnnIter
              .getWordVectors()
              .getWordVector(cnnIter.getWordVectors().vocab().wordAtIndex(0))
              .length;

      final int truncateLength = cnnIter.getTruncateLength();

      if (truncateLength < cl.getKernelSizeX()) {
        throw new InvalidLayerConfigurationException(
            "Kernel row size must be smaller than truncation length. Truncation length was "
                + truncateLength
                + ". Kernel row size was "
                + cl.getKernelSizeX(),
            cl);
      }
      if (truncateLength < cl.getStrideRows()) {
        throw new InvalidLayerConfigurationException(
            "Stride row size must be smaller than truncation length. Truncation length was "
                + truncateLength
                + ". Stride row size was "
                + cl.getStrideColumns(),
            cl);
      }

      if (vectorSize % cl.getKernelSizeY() != 0) {
        throw new InvalidLayerConfigurationException(
            "Wordvector size ("
                + vectorSize
                + ") must be divisible by kernel column size ("
                + cl.getKernelSizeY()
                + ").",
            cl);
      }

      if (vectorSize % cl.getStrideColumns() != 0) {
        throw new InvalidLayerConfigurationException(
            "Wordvector size ("
                + vectorSize
                + ") must be divisible by stride column size ("
                + cl.getStrideColumns()
                + ").",
            cl);
      }

      if (!cl.getConvolutionMode().equals(ConvolutionMode.Same)) {
        throw new InvalidLayerConfigurationException(
            "ConvolutionMode must be ConvolutionMode.Same for ConvolutionLayers in CNN text "
                + "architectures.",
            cl);
      }
    }
  }

  /**
   * Get a peak at the features of the {@code iterator}'s first batch using the given instances.
   *
   * @return Features of the first batch
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
   */
  protected TrainingListener getListener() throws Exception {
    int numSamples = trainData.numInstances();
    TrainingListener listener;

    // Initialize weka listener
    if (iterationListener instanceof weka.dl4j.listener.EpochListener) {
      int numEpochs = getNumEpochs();
      iterationListener
          .init(
              trainData.numClasses(),
              numEpochs,
              numSamples,
              trainIterator,
              earlyStopping.getValDataSetIterator());
      ((EpochListener) iterationListener).setLogFile(logFile);
    }
    listener = iterationListener;
    return listener;
  }

  /**
   * Perform another epoch.
   */
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
      numEpochsPerformed++;
      log.info("Epoch [{}/{}] took {}", numEpochsPerformed, numEpochs, sw.toString());
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

  /**
   * Clean up after learning.
   */
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
    this.zooModel = zooModel;

    ClassLoader origLoader = Thread.currentThread().getContextClassLoader();
    try {
      // Try to parse the layers so the user can change them afterwards
      final int dummyNumLabels = 2;

      Thread.currentThread().setContextClassLoader(this.getClass().getClassLoader());
      ComputationGraph tmpCg = zooModel.init(dummyNumLabels, getSeed(), zooModel.getShape()[0]);
      tmpCg.init();
      layers =
          Arrays.stream(tmpCg.getLayers())
              .map(l -> Layer.create(l.conf().getLayer()))
              .collect(Collectors.toList())
              .toArray(new Layer[tmpCg.getLayers().length]);


    } catch (Exception e) {
      if (!(zooModel instanceof CustomNet)) {
        log.error("Could not set layers from zoomodel.", e);
      }
    } finally {
      Thread.currentThread().setContextClassLoader(origLoader);
    }
  }

  public TrainingListener getIterationListener() {
    return iterationListener;
  }

  @OptionMetadata(
      displayName = "set the iteration listener",
      description = "Set the iteration listener.",
      commandLineParamName = "iteration-listener",
      commandLineParamSynopsis = "-iteration-listener <string>",
      displayOrder = 9
  )
  public void setIterationListener(TrainingListener l) {
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
   * it is numeric)
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
   * (if it is numeric)
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
          int jResorted = fixLabelIndexIfNominal(j, insts);
          preds[i + offset][j] = predBatch.getDouble(i, jResorted);
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
   * Fixes nominal label indices. Dl4j sorts them during training time. A mapping from weka-labels
   * resorted labels is stored in {@link this.labelsSortIndex}.
   *
   * @param j Original index
   * @param insts Test dataset
   * @return Remapped index if test dataset has nominal label. Else return {@code j}
   */
  protected int fixLabelIndexIfNominal(int j, Instances insts) {
    if (insts.classAttribute().isNominal()) {
      return labelSortIndex[j];
    } else {
      return j;
    }
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
   * Get the {@link ComputationGraph} model
   *
   * @return ComputationGraph model
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
    if (model == null || model.getConfiguration() == null) {
      return "";
    }

    final String modelSummary = model.summary();
    final String networkConfigurationString = netConfig.toString();
    final StringBuilder sb = new StringBuilder();
    sb.append("Network Configuration: \n");
    sb.append(networkConfigurationString);
    sb.append("\n");
    sb.append("Model Summary: \n");
    sb.append(modelSummary);
    return sb.toString();
  }

  /**
   * Check if the user has selected to use a zoomodel
   *
   * @return True if zoomodel is not CustomNet
   */
  protected boolean useZooModel() {
    return !(zooModel instanceof CustomNet);
  }

  /**
   * Returns the activations at a certain
   *
   * @param layerName Layer name to get the activations from
   * @return Activations in form of instances
   */
  public Instances getActivationsAtLayer(String layerName, Instances input) throws Exception {
    DataSetIterator iter = getDataSetIterator(input);
    iter.reset();
    DataSet next;
    INDArray acts = null;
    while (iter.hasNext()) {
      next = iter.next();
      INDArray features = next.getFeatures();
      int layerIdx = model.getLayer(layerName).getIndex() - 1;
      Map<String, INDArray> activations = model.feedForward(features, layerIdx, false);
      INDArray activationAtLayer = activations.get(layerName);

      if (acts == null) {
        acts = activationAtLayer;
      } else {
        acts = Nd4j.concat(0, acts, activationAtLayer);
      }
    }

    if (acts == null) {
      return new Instances(input, 0);
    } else {
      return Utils.ndArrayToInstances(acts);
    }
  }
}
