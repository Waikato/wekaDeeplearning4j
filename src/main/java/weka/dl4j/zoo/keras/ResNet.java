package weka.dl4j.zoo.keras;

public class ResNet extends KerasZooModel {
    public enum VARIATION {RESNET50, RESNET50V2, RESNET101, RESNET101V2, RESNET152, RESNET152V2};

    public static int[] inputShape = new int[] {3, 224, 224};

    protected VARIATION m_variation = VARIATION.RESNET50;

    public ResNet() {
        setVariation(VARIATION.RESNET50);
    }

    @Override
    public void setVariation(Enum variation) {
        this.m_variation = (VARIATION) variation;
    }

    @Override
    public String modelFamily() {
        return "keras_resnet";
    }

    @Override
    public String modelPrettyName() {
        switch (m_variation) {
            case RESNET50:
                return "ResNet50";
            case RESNET50V2:
                return "ResNet50V2";
            case RESNET101:
                return "ResNet101";
            case RESNET101V2:
                return "ResNet101V2";
            case RESNET152:
                return "ResNet152";
            case RESNET152V2:
                return "ResNet152V2";
            default:
                return null;
        }
    }
}
