
package weka.dl4j;

/**
 * Cache modes for datasetiterators.
 * <ul>
 *   <li>NONE: Do not use any cache</li>
 *   <li>MEMORY: Cache data in memory</li>
 *   <li>FILESYSTEM: Cache data in the filesystem in "java.io.tmpdir"</li>
 * </ul>
 *
 * @author Steven Lang
 */
public enum CacheMode {
  NONE,
  MEMORY,
  FILESYSTEM
}
