package weka.core;

/** Exception raised in the case of a missing output layer as last layer */
public class MissingOutputLayerException extends WekaException {
  private static final long serialVersionUID = 1038306995981039092L;

  public MissingOutputLayerException(String message) {
    super(message);
  }

  public MissingOutputLayerException(String message, Throwable cause) {
    super(message, cause);
  }
}
