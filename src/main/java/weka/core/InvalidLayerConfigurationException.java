
package weka.core;

import weka.dl4j.layers.Layer;

/**
 * Exception raised when the the configuration is invalid.
 *
 * @author Steven Lang
 */
public class InvalidLayerConfigurationException extends WekaException {

  private static final long serialVersionUID = -8089134131947363159L;

  public InvalidLayerConfigurationException(String message, Layer layer) {
    super("Layer=" + layer.getLayerName() + ": " + message);
  }

  public InvalidLayerConfigurationException(String message, Layer layer, Throwable cause) {
    super("Layer=" + layer.getLayerName() + ": " + message, cause);
  }
}
