package weka.dl4j.zoo.keras;

public class ResNet extends KerasZooModel {
    private static final long serialVersionUID = 5315928260273073018L;

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
                return "KerasResNet50";
            case RESNET50V2:
                return "KerasResNet50V2";
            case RESNET101:
                return "KerasResNet101";
            case RESNET101V2:
                return "KerasResNet101V2";
            case RESNET152:
                return "KerasResNet152";
            case RESNET152V2:
                return "KerasResNet152V2";
            default:
                return null;
        }
    }
}
