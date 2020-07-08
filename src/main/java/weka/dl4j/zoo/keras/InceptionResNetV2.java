package weka.dl4j.zoo.keras;

public class InceptionResNetV2 extends KerasZooModel {
    private static final long serialVersionUID = 7648403112324512260L;

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
