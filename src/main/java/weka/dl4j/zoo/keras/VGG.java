package weka.dl4j.zoo.keras;

public class VGG extends KerasZooModel {
    private static final long serialVersionUID = 3908733587819287902L;

    public enum VARIATION {VGG16, VGG19};

    public static int[] inputShape = new int[] {3, 224, 224};

    protected VARIATION m_variation = VARIATION.VGG16;

    public VGG() {
        setVariation(VARIATION.VGG16);
    }

    @Override
    public void setVariation(Enum variation) {
        this.m_variation = (VARIATION) variation;
    }

    @Override
    public String modelFamily() {
        return "keras_vgg";
    }

    @Override
    public String modelPrettyName() {
        switch (m_variation) {
            case VGG16:
                return "VGG16";
            case VGG19:
                return "VGG19";
            default:
                return null;
        }
    }
}
