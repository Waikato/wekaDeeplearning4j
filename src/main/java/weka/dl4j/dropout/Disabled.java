package weka.dl4j.dropout;

import lombok.EqualsAndHashCode;
import lombok.ToString;

@EqualsAndHashCode(callSuper = true)
@ToString
public class Disabled extends AbstractDropout<org.deeplearning4j.nn.conf.dropout.Dropout> {

  private static final long serialVersionUID = 8082864981844682636L;

  @Override
  public void initializeBackend() {
    backend = null;
  }
}
