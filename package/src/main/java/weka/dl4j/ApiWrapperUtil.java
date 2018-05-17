package weka.dl4j;

import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.util.Set;
import org.reflections.Reflections;

/**
 * This utility class manages loading the appropriate wrapping class for a given backend object
 *
 * @author Steven Lang
 */
public class ApiWrapperUtil {

  /** Return the implementing wrapper class of a given backend object. */
  public static <T extends ApiWrapper<V>, V> T getImplementingWrapper(
      Class<T> cls, V newBackend, String packageName) {
    try {

      Set<Class<? extends T>> classes = getSubclassesOf(cls, packageName);

      for (Class<? extends T> cl : classes) {
        // Skip abstract classes
        if (Modifier.isAbstract(cl.getModifiers())) {
          continue;
        }
        final Class<? extends V> backendClass = getBackendClass(cl);

        if (backendClass.equals(newBackend.getClass())) {
          return instantiateBackend(newBackend, cl);
        }
      }

      throw new RuntimeException(
          String.format(
              "Could not find any class that implements %s and wraps %s in package '%s'.",
              cls.getSimpleName(), newBackend.getClass().getSimpleName(), packageName));

    } catch (IllegalAccessException | InstantiationException e) {
      e.printStackTrace();
      throw new RuntimeException(
          String.format(
              "Failed finding a class that implements %s and wraps %s in package '%s' with exception: %s",
              cls.getSimpleName(),
              newBackend.getClass().getSimpleName(),
              packageName,
              e.toString()));
    }
  }

  /** Instantiate the given backend class. */
  private static <T extends ApiWrapper<V>, V> T instantiateBackend(
      V newBackend, Class<? extends T> cl) throws InstantiationException, IllegalAccessException {
    T obj = cl.newInstance();
    obj.setBackend(newBackend);
    return obj;
  }

  /** Given a wrapping class, get the backend (to be wrapped) class. */
  private static <T extends ApiWrapper<V>, V> Class<? extends V> getBackendClass(
      Class<? extends T> cl) {
    return (Class<? extends V>)
        ((ParameterizedType) cl.getGenericSuperclass()).getActualTypeArguments()[0];
  }

  /** Get all subclasses of the given class. */
  private static <T extends ApiWrapper<V>, V> Set<Class<? extends T>> getSubclassesOf(
      Class<T> cls, String packageName) {
    Reflections reflections = new Reflections(packageName);
    return reflections.getSubTypesOf(cls);
  }
}
