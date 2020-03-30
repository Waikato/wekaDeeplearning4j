package weka.dl4j.zoo.keras;

// TODO check all models download properly

public class MobileNet extends KerasZooModel {
    public enum VARIATION {
        V1,
        V2
    };

    public static int[] inputShape = new int[] {3, 224, 224};

    protected VARIATION m_variation = VARIATION.V2;

    public MobileNet() {
        setVariation(VARIATION.V2);
    }

    @Override
    public void setVariation(Enum variation) {
        this.m_variation = (VARIATION) variation;
    }

    @Override
    public String modelFamily() {
        return "keras_mobilenet";
    }

    @Override
    public String modelPrettyName() {
        switch (m_variation) {
            case V1:
                return "MobileNet";
            case V2:
                return "MobileNetV2";
            default:
                return null;
        }
    }
}
