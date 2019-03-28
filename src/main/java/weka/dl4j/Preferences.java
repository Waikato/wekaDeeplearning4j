
package weka.dl4j;

import org.deeplearning4j.nn.conf.WorkspaceMode;

/**
 * Preferences class for Deeplearning4j/Nd4j static settings that should be used across the
 * package.
 *
 * @author Steven Lang
 */
public class Preferences {

  /**
   * Global workspace mode
   */
  public static WorkspaceMode WORKSPACE_MODE = WorkspaceMode.ENABLED;
}
