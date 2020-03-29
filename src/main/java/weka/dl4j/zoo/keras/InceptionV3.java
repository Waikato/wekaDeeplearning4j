package weka.dl4j.zoo.keras;

// TODO check all models download properly

public class InceptionV3 extends KerasZooModel {
    public enum VARIATION {
        STANDARD
    };

    public static int[] inputShape = new int[] {3, 299, 299}; //TODO ensure that ImageInstanceIterator adheres to this

    protected VARIATION m_variation = VARIATION.STANDARD;

    public InceptionV3() {
        setVariation(VARIATION.STANDARD);
    }

    @Override
    public void setVariation(Enum variation) {
        this.m_variation = (VARIATION) variation;
    }

    @Override
    public String modelFamily() {
        return "keras_inceptionv3";
    }

    @Override
    public String modelPrettyName() {
        return "InceptionV3";
    }
}
