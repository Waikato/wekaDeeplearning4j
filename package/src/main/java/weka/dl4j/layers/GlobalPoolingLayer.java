package weka.dl4j.layers;

import org.deeplearning4j.nn.conf.layers.PoolingType;
import weka.core.OptionMetadata;

/**
 * A version of DeepLearning4j's GlobalPooling that implements WEKA option handling.
 *
 * @author Steven Lang
 */
public class GlobalPoolingLayer extends org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer {

  private static final long serialVersionUID = 2882286002911860559L;
  /** Constructor for setting some defaults. */
  public GlobalPoolingLayer() {
    setLayerName("GlobalPooling layer");
    setPoolingType(PoolingType.MAX);
    setPnorm(2);
  }

  @OptionMetadata(
    displayName = "pooling type",
    description = "The type of pooling to use (default = MAX; options: MAX, AVG, SUM, NONE).",
    commandLineParamName = "poolingType",
    commandLineParamSynopsis = "-poolingType <string>",
    displayOrder = 10
  )
  @Override
  public PoolingType getPoolingType() {
    return super.getPoolingType();
  }

  @Override
  public void setPoolingType(PoolingType poolingType) {
    super.setPoolingType(poolingType);
  }

  @OptionMetadata(
    displayName = "pooling dimensions",
    description = "The pooling dimensions (default = [2,2]).",
    commandLineParamName = "poolDimensions",
    commandLineParamSynopsis = "-poolDimensions <int>",
    displayOrder = 4
  )
  @Override
  public int[] getPoolingDimensions() {
    return super.getPoolingDimensions();
  }

  @Override
  public void setPoolingDimensions(int[] poolingDimensions) {
    super.setPoolingDimensions(poolingDimensions);
  }

  @OptionMetadata(
    displayName = "pnorm",
    description = "The value of the pnorm parameter (default = 2).",
    commandLineParamName = "pnorm",
    commandLineParamSynopsis = "-pnorm <int>",
    displayOrder = 3
  )
  @Override
  public int getPnorm() {
    return super.getPnorm();
  }

  @Override
  public void setPnorm(int pnorm) {
    super.setPnorm(pnorm);
  }

  @OptionMetadata(
    displayName = "collapse dimensions",
    description = "Wether to collapse dimensions (default = true).",
    commandLineParamName = "collapseDimensions",
    commandLineParamSynopsis = "-collapseDimensions <boolean>",
    displayOrder = 11
  )
  @Override
  public boolean isCollapseDimensions() {
    return super.isCollapseDimensions();
  }

  @Override
  public void setCollapseDimensions(boolean collapseDimensions) {
    super.setCollapseDimensions(collapseDimensions);
  }

  @OptionMetadata(
    displayName = "layer name",
    description = "The name of the layer (default = GlobalPooling Layer).",
    commandLineParamName = "name",
    commandLineParamSynopsis = "-name <string>",
    displayOrder = 0
  )
  @Override
  public String getLayerName() {
    return super.getLayerName();
  }

  @Override
  public void setLayerName(String layerName) {
    super.setLayerName(layerName);
  }

  @OptionMetadata(
    displayName = "dropout parameter",
    description = "The dropout parameter (default = 0).",
    commandLineParamName = "dropout",
    commandLineParamSynopsis = "-dropout <double>",
    displayOrder = 11
  )
  @Override
  public double getDropOut() {
    return super.getDropOut();
  }

  @Override
  public void setDropOut(double dropOut) {
    super.setDropOut(dropOut);
  }

  /**
   * Global info.
   *
   * @return string describing this class.
   */
  public String globalInfo() {
    return "A global pooling layer from DeepLearning4J.";
  }
}
