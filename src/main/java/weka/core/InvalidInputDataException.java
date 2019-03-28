
package weka.core;

/** Exception raised in the case of an invalid input data
 *
 * @author Steven Lang
 */
public class InvalidInputDataException extends WekaException {

  private static final long serialVersionUID = -388372062851727428L;

  public InvalidInputDataException(String message) {
    super(message);
  }

  public InvalidInputDataException(String message, Throwable cause) {
    super(message, cause);
  }
}
