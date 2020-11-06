package weka.dl4j.inference;

public class PredictionClass {
    /**
     * ID of the class prediction
     */
    private final int classID;

    /**
     * Human readable class name
     */
    private final String className;

    public PredictionClass(int classID, String className) {
        this.classID = classID;
        this.className = className;
    }

    public int getClassID() {
        return classID;
    }

    public String getClassName() {
        return className;
    }

    @Override
    public String toString() {
        return String.format("Class ID = %d, Class name = %s",
                getClassID(),
                getClassName());
    }
}
