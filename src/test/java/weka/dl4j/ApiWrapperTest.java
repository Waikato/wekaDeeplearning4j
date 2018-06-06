package weka.dl4j;

import org.junit.Before;

/**
 * An abstract test class for api wrapper objects.
 *
 * @param <T> Implementing api wrapper class
 * @author Steven Lang
 */
public abstract class ApiWrapperTest<T extends ApiWrapper> {

  protected T wrapper;

  /**
   * Double comparision precision
   */
  protected double PRECISION = 1e-7;

  public abstract T getApiWrapper();

  @Before
  public void setup(){
    wrapper = getApiWrapper();
  }

}
