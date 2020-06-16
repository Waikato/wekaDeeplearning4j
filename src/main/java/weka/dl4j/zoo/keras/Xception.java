package weka.dl4j.zoo.keras;

public class Xception extends KerasZooModel {
    private static final long serialVersionUID = -5423178061075685025L;

    public enum VARIATION {STANDARD};

    public static int[] inputShape = new int[] {3, 229, 229};

    protected VARIATION m_variation = VARIATION.STANDARD;

    public Xception() {
        setVariation(VARIATION.STANDARD);
    }

    @Override
    public void setVariation(Enum variation) {
        this.m_variation = (VARIATION) variation;
    }

    @Override
    public String modelFamily() {
        return "keras_xception";
    }

    @Override
    public String modelPrettyName() {
        switch (m_variation) {
            case STANDARD:
                return "KerasXception";
            default:
                return null;
        }
    }
}
