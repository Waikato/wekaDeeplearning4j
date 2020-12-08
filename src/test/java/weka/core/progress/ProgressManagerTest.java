package weka.core.progress;

import junit.framework.TestCase;

public class ProgressManagerTest extends TestCase {

    public void testCheckIfRunByGUI() {
        // Arrange
        ProgressManager progressManager = new ProgressManager();

        // Act/Assert
        assertFalse(progressManager.checkIfRunByGUI());
    }
}