package weka.dl4j;

/**
 * A general interface to access a backend object of a certain deeplearning4j class which is wrapped
 * in a class to make it usable in weka.
 *
 * @param <T> Class which is to be wrapped.
 */
public interface ApiWrapper<T> {

  T getBackend();

  void setBackend(T newBackend);

  void initializeBackend();
}
