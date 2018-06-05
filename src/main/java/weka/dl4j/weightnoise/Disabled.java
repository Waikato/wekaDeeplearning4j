package weka.dl4j.weightnoise;

import java.io.Serializable;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import weka.core.OptionHandler;

/**
 * Disabled option for WeightNoise.
 *
 * @author Steven Lang
 */
@EqualsAndHashCode(callSuper = true)
@ToString
public class Disabled extends AbstractWeightNoise implements OptionHandler, Serializable {

  private static final long serialVersionUID = 4568626187488846101L;

  @Override
  public void initializeBackend() {
    backend=null;
  }

}
