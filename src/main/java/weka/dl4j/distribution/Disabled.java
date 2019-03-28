
package weka.dl4j.distribution;

/**
 * Disabled Distribution.
 *
 * @author Steven Lang
 */
public class Disabled extends Distribution<org.deeplearning4j.nn.conf.distribution.Distribution> {
  private static final long serialVersionUID = -3673597910434423693L;

  @Override
  public void initializeBackend() {
    backend = null;
  }
}
