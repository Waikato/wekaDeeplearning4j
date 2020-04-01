package weka.dl4j.zoo.keras;

public class InceptionResNetV2 extends KerasZooModel {
    public enum VARIATION {
        STANDARD
    };

    public static int[] inputShape = new int[] {3, 299, 299};

    protected VARIATION m_variation = VARIATION.STANDARD;

    public InceptionResNetV2() {
        setVariation(VARIATION.STANDARD);
    }

    @Override
    public void setVariation(Enum variation) {
        this.m_variation = (VARIATION) variation;
    }

    @Override
    public String modelFamily() {
        return "keras_inceptionresnetv2";
    }

    @Override
    public String modelPrettyName() {
        return "InceptionResNetV2";
    }
}
