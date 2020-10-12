package weka.dl4j.inference;

public class PredictionClass {
    /**
     * ID of the class prediction
     */
    protected int classID;

    /**
     * Human readable class name
     */
    protected String className;

    public PredictionClass(int classID, String className) {
        this.classID = classID;
        this.className = className;
    }

    public int getID() {
        return classID;
    }

    public String getName() {
        return className;
    }

    @Override
    public String toString() {
        return String.format("Class ID = %d, Class name = %s",
                getID(),
                getName());
    }
}
