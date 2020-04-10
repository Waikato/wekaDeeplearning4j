/*
 * WekaDeeplearning4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * WekaDeeplearning4j is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with WekaDeeplearning4j.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Dl4jMlpFilter.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.filters.unsupervised.attribute;

import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.Enumeration;

import org.apache.commons.lang.ArrayUtils;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.*;
import weka.dl4j.PoolingType;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.zoo.AbstractZooModel;
import weka.dl4j.zoo.ResNet50;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;

/**
 * Weka filter that uses a neural network trained via {@link Dl4jMlpClassifier} as feature
 * transformation.
 *
 * @author Steven Lang
 */
public class Dl4jMlpFilter extends SimpleBatchFilter implements OptionHandler {

  private static final long serialVersionUID = 21054165308223L;

  /**
   * The classifier model this filter is based on.
   */
  protected File serializedModelFile = new File(WekaPackageManager.getPackageHome().toURI());

  /**
   * The zoo model to use, if we're not loading from the serialized model file
   */
  protected AbstractZooModel zooModelType = new ResNet50();

  /**
   * The image instance iterator to use
   */
  protected ImageInstanceIterator imageInstanceIterator = new ImageInstanceIterator();

  /**
   * Is the supplied dataset a meta file (simply contains locations of image files)
   * or does the arff contain all the image data inside
   */
  protected boolean isMetaArff = true;

  /**
   * Should we use the selected Model Zoo model, or should we use the serialized model
   * as specified by specifiedModelFile
   */
  protected boolean useZooModel = true;

  protected PoolingType poolingType = PoolingType.MAX;

  public PoolingType getPoolingType() {
    return poolingType;
  }

  public void setPoolingType(PoolingType poolingType) {
    this.poolingType = poolingType;
  }

  protected Dl4jMlpClassifier model;

  /**
   * Layer index of the layer which is used to get the outputs from.
   */
  protected String[] transformationLayerNames = new String[] { };

  @OptionMetadata(
      description = "Layer names of the layer used for the feature transformation",
      displayName = "Feature extraction layer",
      commandLineParamName = "layerName",
      commandLineParamSynopsis = "-layerName <String>",
      displayOrder = 0
  )
  public String[] getTransformationLayerNames() {
    return transformationLayerNames;
  }

  public void setTransformationLayerNames(String[] transformationLayerNames) {
    this.transformationLayerNames = transformationLayerNames;
  }

  public void addTransformationLayerName(String transformationLayerName) {
    int n = this.transformationLayerNames.length;
    String[] newArr = new String[n + 1];
    for (int i = 0; i < n; i++)
      newArr[i] = this.transformationLayerNames[i];

    newArr[n] = transformationLayerName;
    this.transformationLayerNames = newArr;
  }

  @OptionMetadata(
      description = "The trained Dl4jMlpClassifier object that contains the network, used for transformation.",
      displayName = "Serialized model file",
      commandLineParamName = "model",
      commandLineParamSynopsis = "-model <File>",
      displayOrder = 1
  )
  public File getSerializedModelFile() {
    return serializedModelFile;
  }

  public void setSerializedModelFile(File modelPath) {
    this.serializedModelFile = modelPath;
  }

  @OptionMetadata(
          description = "The pretrained model from the DL4J Model Zoo (or a keras model)",
          displayName = "Pretrained zoo model",
          commandLineParamName = "zooModel",
          commandLineParamSynopsis = "-zooModel <Model Zoo specification>",
          displayOrder = 2
  )
  public AbstractZooModel getZooModelType() { return zooModelType; }

  public void setZooModelType(AbstractZooModel zooModelType) { this.zooModelType = zooModelType; }

  @OptionMetadata(
          description = "Are the supplied instances 'meta instances' (just point to image file location)",
          displayName = "Using 'meta instances'?",
          commandLineParamName = "isMeta",
          commandLineParamSynopsis = "-isMeta <true|false>",
          displayOrder = 3
  )
  public boolean isMetaArff() { return isMetaArff; }

  public void setMetaArff(boolean metaArff) { isMetaArff = metaArff; }

  @OptionMetadata(
          description = "Use the zoo model specification instead of the serialized model file",
          displayName = "Use the zoo model",
          commandLineParamName = "isZoo",
          commandLineParamSynopsis = "-isZoo <true|false>",
          displayOrder = 4
  )
  public boolean isUseZooModel() { return useZooModel; }

  public void setUseZooModel(boolean useZooModel) { this.useZooModel = useZooModel; }

  @OptionMetadata(
          description = "The instance iterator to use.",
          displayName = "instance iterator", commandLineParamName = "iterator",
          commandLineParamSynopsis = "-iterator <string>"
  )
  public ImageInstanceIterator getImageInstanceIterator() {
    return imageInstanceIterator;
  }

  public void setImageInstanceIterator(ImageInstanceIterator imageInstanceIterator) {
    this.imageInstanceIterator = imageInstanceIterator;
  }

  @Override
  public String globalInfo() {
    return null;
  }

  @Override
  public boolean allowAccessToFullInputFormat() {
    return true;
  }

  private void loadModel(Instances data) throws Exception {
    if (!useZooModel) {
      // First try load from the WEKA binary model file
      try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(serializedModelFile))) {
        model = (Dl4jMlpClassifier) ois.readObject();
      } catch (Exception e) {
        System.err.println("Couldn't load from model file, trying a pretrained zoo model");
        e.printStackTrace();
      }
    } else {
      // If that fails, try loading from selected zoo model (or keras file)
      model = new Dl4jMlpClassifier();
      model.setFilterMode(true);
      model.setZooModel(zooModelType);
      model.setInstanceIterator(imageInstanceIterator);
      model.initializeClassifier(data);

      addTransformationLayerName(zooModelType.getFeatureExtractionLayer());
    }
  }

  @Override
  protected Instances determineOutputFormat(Instances inputFormat) throws Exception {
    loadModel(inputFormat);
    return model.getActivationsAtLayers(transformationLayerNames, inputFormat, poolingType);
  }


  @Override
  protected Instances process(Instances instances) throws Exception {
    return model.getActivationsAtLayers(transformationLayerNames, instances, poolingType);
  }


  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration<Option> listOptions() {
    return Option.listOptionsForClassHierarchy(this.getClass(), Filter.class).elements();
  }

  /**
   * Gets the current settings of the Classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String[] getOptions() {
    return Option.getOptionsForHierarchy(this, Filter.class);
  }

  /**
   * Parses a given list of options.
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {
    Option.setOptionsForHierarchy(options, this, Filter.class);
  }
}
