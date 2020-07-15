package weka.dl4j.playground;

public class Prediction {

    private int classID;

    private String className;

    private double classProbability;

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
