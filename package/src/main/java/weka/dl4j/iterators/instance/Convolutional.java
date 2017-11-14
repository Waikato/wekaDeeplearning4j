package weka.dl4j.iterators.instance;

/**
 * Interface for objects for which convolution can be applied.
 *
 * @author Steven Lang
 */
public interface Convolutional {

    int getHeight();

    int getWidth();

    int getNumChannels();
}
