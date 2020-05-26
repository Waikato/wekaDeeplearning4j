package weka.dl4j.zoo.keras;

import org.deeplearning4j.zoo.ModelMetaData;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DenseNet extends KerasZooModel {
    private static final long serialVersionUID = 982160047744739299L;

    public enum VARIATION {DENSENET121, DENSENET169, DENSENET201};

    public static int[] inputShape = new int[] {3, 224, 224};

    protected VARIATION m_variation = VARIATION.DENSENET121;

    public DenseNet() {
        setVariation(VARIATION.DENSENET121);
    }

    @Override
    public void setVariation(Enum variation) {
        this.m_variation = (VARIATION) variation;
    }

    @Override
    public String modelFamily() {
        return "keras_densenet";
    }

    @Override
    public String modelPrettyName() {
        switch (m_variation) {
            case DENSENET121:
                return "KerasDenseNet121";
            case DENSENET169:
                return "KerasDenseNet169";
            case DENSENET201:
                return "KerasDenseNet201";
            default:
                return null;
        }
    }
}
