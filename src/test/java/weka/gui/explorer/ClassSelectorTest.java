package weka.gui.explorer;

import junit.framework.TestCase;
import weka.dl4j.inference.ClassmapType;
import weka.dl4j.inference.ModelOutputDecoder;
import weka.dl4j.inference.PredictionClass;
import weka.dl4j.inference.TopNPredictions;

import java.util.ArrayList;

public class ClassSelectorTest extends TestCase {

    /**
     * Test that we get the expected classes from passing a regex in
     */
    public void testGetMatchingClasses() {
        // Arrange
        ModelOutputDecoder decoder = new ModelOutputDecoder();
        decoder.setBuiltInClassMap(ClassmapType.IMAGENET);
        ClassSelector classSelector = new ClassSelector(null, decoder.getClasses());
        int sportsCarID = 817;

        // Act
        ArrayList<PredictionClass> predictionClasses = classSelector.getMatchingClasses(".*car.*");

        // Assert
        assertEquals(31, predictionClasses.size());
        assertTrue(predictionClasses.stream().anyMatch(x -> x.getClassID() == sportsCarID));
    }

    /**
     * Test that we get the expected classes from passing a regex in
     */
    public void testNoMatchingClasses_ForEmptyPattern() {
        // Arrange
        ModelOutputDecoder decoder = new ModelOutputDecoder();
        decoder.setBuiltInClassMap(ClassmapType.IMAGENET);
        ClassSelector classSelector = new ClassSelector(null, decoder.getClasses());

        // Act
        ArrayList<PredictionClass> predictionClasses = classSelector.getMatchingClasses("");

        // Assert
        assertEquals(0, predictionClasses.size());
    }
}