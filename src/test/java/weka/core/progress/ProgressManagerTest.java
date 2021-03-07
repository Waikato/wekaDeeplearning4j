package weka.core.progress;

import junit.framework.TestCase;

/**
 * Tests for the ProgressManager class.
 */
public class ProgressManagerTest extends TestCase {

    /**
     * Check that the IfRunByGUI flag correctly returns false when run from code.
     */
    public void testCheckIfRunByGUI() {
        // Arrange
        ProgressManager progressManager = new ProgressManager();

        // Act/Assert
        assertFalse(progressManager.checkIfRunByGUI());
    }
}