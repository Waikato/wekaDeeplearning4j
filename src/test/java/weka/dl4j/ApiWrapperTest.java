
package weka.dl4j;

import java.util.Arrays;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import weka.core.OptionHandler;

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

  @Test
  public void testOptions() throws Exception {
    if (wrapper instanceof OptionHandler){
      OptionHandler optionHandler = ((OptionHandler) wrapper);
      String[] options = optionHandler.getOptions();
      String[] optionsCopy = Arrays.copyOf(options, options.length);
      optionHandler.setOptions(optionsCopy);
      String[] optionsAfter = optionHandler.getOptions();

      Assert.assertArrayEquals(options, optionsAfter);
    }
  }

  @Test
  public void testSettingBackends(){
    T apiWrapper1 = getApiWrapper();
    T apiWrapper2 = getApiWrapper();

    apiWrapper2.setBackend(apiWrapper1.getBackend());

    Assert.assertEquals(apiWrapper1, apiWrapper2);
    Assert.assertEquals(apiWrapper1.getBackend(), apiWrapper2.getBackend());
  }

}
