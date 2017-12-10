package weka.classifiers.functions;

import java.io.Serializable;
import java.util.Arrays;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.AbstractLSTM;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.MissingOutputLayerException;
import weka.core.OptionHandler;
import weka.core.OptionMetadata;
import weka.dl4j.CacheMode;
import weka.dl4j.iterators.instance.sequence.AbstractSequenceInstanceIterator;
import weka.dl4j.iterators.instance.sequence.text.TextEmbeddingInstanceIterator;

/**
 * A classifier that can handle sequences.
 *
 * @author Steven Lang
 */
@Slf4j
public class RnnSequenceClassifier extends Dl4jMlpClassifier
    implements CapabilitiesHandler, Classifier, Serializable, OptionHandler {
  /** SerialVersionUID */
  private static final long serialVersionUID = 5643486590174837865L;

  /** Truncated backpropagation through time backward length */
  private int tBPTTbackwardLength = 25;
  /** Truncated backpropagation through time forward length */
  private int tBPTTforwardLength = 25;

  public RnnSequenceClassifier() {
    super();
    layers = new Layer[] {new weka.dl4j.layers.RnnOutputLayer()};
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
    if (layers.length == 0) {
      throw new MissingOutputLayerException("No layers have been added!");
    }

    final Layer lastLayer = layers[layers.length - 1];
    if (!(lastLayer instanceof RnnOutputLayer)) {
      throw new MissingOutputLayerException("Last layer in network must be an output layer!");
    }

    ClassLoader origLoader = Thread.currentThread().getContextClassLoader();
    try {
      Thread.currentThread().setContextClassLoader(this.getClass().getClassLoader());

      data = initEarlyStopping(data);
      this.trainData = data;

      instanceIterator.initialize();

      createModel();

      // Setup the datasetiterators (needs to be done after the model initialization)
      trainIterator = getDataSetIterator(this.trainData);

      // Set the iteration listener
      model.setListeners(getListener());

      numEpochsPerformed = 0;
    } finally {
      Thread.currentThread().setContextClassLoader(origLoader);
    }
  }

  @Override
  protected Instances applyFilters(Instances insts) throws Exception {
    // No filters currently
    return insts;
  }

  @Override
  protected void createModel() throws Exception {
    final INDArray features = getFirstBatchFeatures(trainData);
    log.info("Feature shape: {}", features.shape());
    ComputationGraphConfiguration.GraphBuilder gb =
        netConfig
            .builder()
            .seed(getSeed())
            .inferenceWorkspaceMode(WorkspaceMode.SEPARATE)
            .trainingWorkspaceMode(WorkspaceMode.SEPARATE)
            .graphBuilder()
            .backpropType(BackpropType.TruncatedBPTT)
            .tBPTTBackwardLength(tBPTTbackwardLength)
            .tBPTTForwardLength(tBPTTforwardLength);

    // Set ouput size
    final Layer lastLayer = layers[layers.length - 1];
    final int nOut = trainData.numClasses();
    if (lastLayer instanceof RnnOutputLayer) {
      ((RnnOutputLayer) lastLayer).setNOut(nOut);
    }

    String currentInput = "input";
    gb.addInputs(currentInput);
    // Collect layers
    for (Layer layer : layers) {
      String lName = layer.getLayerName();
      gb.addLayer(lName, layer, currentInput);
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
   * Validate if the given layers are compatible with this classifier.
   *
   * @param layers New set of layers
   */
  @Override
  protected void validateLayers(Layer[] layers) {
    final boolean valid = Arrays.stream(layers).allMatch(this::isSequenceCompatibleLayer);
    if (!valid) {
      throw new RuntimeException(
          "You have chosen an unsupported layer type. Pick one of "
              + "[EmeddingLayer, LSTM, GravesLSTM, RNNOutput].");
    }
  }

  /**
   * Check if the given layers are compatible for sequences (Only allow embedding and RNN for now)
   *
   * @param layer Layers to check
   * @return True if compatible
   */
  private boolean isSequenceCompatibleLayer(Layer layer) {
    return layer instanceof EmbeddingLayer
        || layer instanceof AbstractLSTM
        || layer instanceof RnnOutputLayer;
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

    log.info("Calc. dist for {} instances", insts.numInstances());

    // Do we only have a ZeroR model?
    if (zeroR != null) {
      return zeroR.distributionsForInstances(insts);
    }

    // Process input data to have the same filters applied as the training data
    insts = applyFilters(insts);

    // Get predictions
    final DataSetIterator it = getDataSetIterator(insts, CacheMode.NONE);
    double[][] preds = new double[insts.numInstances()][insts.numClasses()];

    if (it.resetSupported()) {
      it.reset();
    }

    int offset = 0;
    boolean next = it.hasNext();

    // Get predictions batch-wise
    while (next) {
      final DataSet ds = it.next();
      final INDArray features = ds.getFeatureMatrix();
      final INDArray labelsMask = ds.getLabelsMaskArray();
      INDArray lastTimeStepIndices = Nd4j.argMax(labelsMask, 1);
      INDArray predBatch = model.outputSingle(features);
      int currentBatchSize = predBatch.size(0);
      for (int i = 0; i < currentBatchSize; i++) {
        int thisTimeSeriesLastIndex = lastTimeStepIndices.getInt(i);
        INDArray thisExampleProbabilities =
            predBatch.get(
                NDArrayIndex.point(i),
                NDArrayIndex.all(),
                NDArrayIndex.point(thisTimeSeriesLastIndex));
        for (int j = 0; j < insts.numClasses(); j++) {
          preds[i + offset][j] = thisExampleProbabilities.getDouble(j);
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

  @Override
  public double[] distributionForInstance(Instance instance) throws Exception {

    Instances data = new Instances(instance.dataset());
    data.add(instance);
    return distributionsForInstances(data)[0];
  }

  @OptionMetadata(
    description = "Number of backpropagations through time backward (default = 25).",
    displayName = "truncated backprop through time backward",
    commandLineParamName = "tBPTTBackward",
    commandLineParamSynopsis = "-tBPTTBackward <int>",
    displayOrder = 20
  )
  public int gettBPTTbackwardLength() {
    return tBPTTbackwardLength;
  }

  public void settBPTTbackwardLength(int tBPTTbackwardLength) {
    this.tBPTTbackwardLength = tBPTTbackwardLength;
  }

  @OptionMetadata(
    description = "Number of backpropagations through time forward (default = 25).",
    displayName = "truncated backprop through time forward",
    commandLineParamName = "tBPTTForward",
    commandLineParamSynopsis = "-tBPTTForward <int>",
    displayOrder = 21
  )
  public int gettBPTTforwardLength() {
    return tBPTTforwardLength;
  }

  public void settBPTTforwardLength(int tBPTTforwardLength) {
    this.tBPTTforwardLength = tBPTTforwardLength;
  }

  public void setInstanceIterator(AbstractSequenceInstanceIterator iterator) {
    instanceIterator = iterator;
  }

  /**
   * Returns default capabilities of the classifier.
   *
   * @return the capabilities of this classifier
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    result.enable(Capabilities.Capability.STRING_ATTRIBUTES);

    // class
    result.enable(Capabilities.Capability.NOMINAL_CLASS);
    result.enable(Capabilities.Capability.NUMERIC_CLASS);
    result.enable(Capabilities.Capability.DATE_CLASS);
    result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

    return result;
  }

  public String globalInfo() {
    return "Classification and regression with recurrent neural network using DeepLearning4J.\n"
        + "Iterator usage\n"
        + "- TextEmbeddingInstanceIterator: ARFF files containing the document as lines\n"
        + "- TextFilesEmbeddingInstanceIterator: ARFF files containing meta-data linking to actual document files\n"
        + "(See also https://deeplearning.cms.waikato.ac.nz/user-guide/data/ )";
  }

}
