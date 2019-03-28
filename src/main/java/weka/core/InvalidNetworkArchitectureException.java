
package weka.core;

/**
 * Exception raised when the the network architecture is invalid.
 *
 * @author Steven Lang
 */
public class InvalidNetworkArchitectureException extends WekaException {


  private static final long serialVersionUID = 4688147827308527299L;

  public InvalidNetworkArchitectureException(String message) {
    super(message);
  }

  public InvalidNetworkArchitectureException(String message, Throwable cause) {
    super(message, cause);
  }
}
