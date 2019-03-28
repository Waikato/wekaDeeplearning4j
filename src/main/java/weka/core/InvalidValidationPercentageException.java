
package weka.core;

/**
 * Exception raised in the case of a validation percentage which might be too low / too high
 *
 * @author Steven Lang
 */
public class InvalidValidationPercentageException extends WekaException {

  private static final long serialVersionUID = 7595561274022725092L;

  public InvalidValidationPercentageException(String message) {
    super(message);
  }

  public InvalidValidationPercentageException(String message, Throwable cause) {
    super(message, cause);
  }
}
