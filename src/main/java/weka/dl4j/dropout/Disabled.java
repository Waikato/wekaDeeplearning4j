package weka.dl4j.dropout;

import lombok.EqualsAndHashCode;
import lombok.ToString;
import org.deeplearning4j.nn.conf.dropout.Dropout;
import weka.dl4j.dropout.Disabled.DisabledDropoutImpl;

@EqualsAndHashCode(callSuper = true)
@ToString
public class Disabled extends AbstractDropout<DisabledDropoutImpl> {

  private static final long serialVersionUID = 8082864981844682636L;

  @Override
  public void initializeBackend() {
    backend = null;
  }

  /**
   * Dummy dropout implementation.
   */
  protected class DisabledDropoutImpl extends Dropout{
    private static final long serialVersionUID = 5933930276882455322L;
    public DisabledDropoutImpl(double activationRetainProbability) {
      super(activationRetainProbability);
    }
  }
}
