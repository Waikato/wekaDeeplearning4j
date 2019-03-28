
package weka.core;

/**
 * Exception raised in the case of a wrong iterator
 *
 * @author Steven Lang
 */
public class WrongIteratorException extends WekaException {

  private static final long serialVersionUID = 1038306995981039092L;

  public WrongIteratorException(String message) {
    super(message);
  }

  public WrongIteratorException(String message, Throwable cause) {
    super(message, cause);
  }
}
