package weka.dl4j.playground;

/**
 * Simple class to hold the necessary values for prediction
 * @author - Rhys Compton
 */
public class Prediction {

    /**
     * ID of the class prediction
     */
    protected int classID;

    /**
     * Human readable class name
     */
    protected String className;

    /**
     * Probability of the predicted class
     */
    protected double classProbability;

    public Prediction(int classID, String className, double classProbability) {
        this.classID = classID;
        this.className = className;
        this.classProbability = classProbability;
    }

    public int getClassID() {
        return classID;
    }

    public String getClassName() {
        return className;
    }

    public double getClassProbability() {
        return classProbability;
    }

    @Override
    public String toString() {
        return String.format("Class ID = %d, Class name = %s, Class prob = %.3f",
                getClassID(),
                getClassName(),
                getClassProbability());
    }
}
