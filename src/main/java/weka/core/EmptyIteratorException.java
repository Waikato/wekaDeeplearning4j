
package weka.core;

/**
 * Exception raised when the iterator was unexpectedly empty
 *
 * @author Steven Lang
 */
public class EmptyIteratorException extends WekaException {

  private static final long serialVersionUID = 7159773687653762115L;

  public EmptyIteratorException(String message) {
    super(message);
  }

  public EmptyIteratorException(String message, Throwable cause) {
    super(message, cause);
  }
}
