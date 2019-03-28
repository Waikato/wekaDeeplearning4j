
package weka.dl4j;

/**
 * A general interface to access a backend object of a certain deeplearning4j class which is wrapped
 * in a class to make it usable in weka.
 *
 * @param <T> Class which is to be wrapped.
 * @author Steven Lang
 */
public interface ApiWrapper<T> {

    /**
     * Access the DL4J backend.
     *
     * @return DL4J backend for this wrapper
     */
    T getBackend();

    /**
     * Set the DL4J backend.
     *
     * @param newBackend Backend that should be wrapped by this class
     */
    void setBackend(T newBackend);

    /**
     * Initialize the DL4J backend.
     */
    void initializeBackend();
}
