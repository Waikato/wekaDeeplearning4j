package weka.dl4j.zoo.keras;

public class NASNet extends KerasZooModel {
    private static final long serialVersionUID = -8106239516237343141L;

    public enum VARIATION {
        MOBILE,
        LARGE
    };

    public static int[] inputShape = new int[] {3, 224, 224};

    protected VARIATION m_variation = VARIATION.MOBILE;

    public NASNet() {
        setVariation(VARIATION.MOBILE);
    }

    @Override
    public void setVariation(Enum variation) {
        this.m_variation = (VARIATION) variation;

        if (this.m_variation == VARIATION.MOBILE) {
            inputShape[1] = 224;
            inputShape[2] = 224;
        } else if (this.m_variation == VARIATION.LARGE) {
            inputShape[1] = 331;
            inputShape[2] = 331;
        }
    }

    @Override
    public String modelFamily() {
        return "keras_nasnet";
    }

    @Override
    public String modelPrettyName() {
        switch (m_variation) {
            case MOBILE:
                return "KerasNASNetMobile";
            case LARGE:
                return "KerasNASNetLarge";
            default:
                return null;
        }
    }
}
