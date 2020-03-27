package weka.dl4j.zoo.keras;

import org.deeplearning4j.zoo.ModelMetaData;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class EfficientNet extends KerasZooModel {
    public enum VARIATION {
        EFFICIENTNET_B0,
        EFFICIENTNET_B1,
        EFFICIENTNET_B2,
        EFFICIENTNET_B3,
        EFFICIENTNET_B4,
        EFFICIENTNET_B5,
        EFFICIENTNET_B6,
        EFFICIENTNET_B7,
    };

    public static int[] inputShape = new int[] {3, 224, 224};

    protected VARIATION m_variation = VARIATION.EFFICIENTNET_B0;

    public EfficientNet() {
        setVariation(VARIATION.EFFICIENTNET_B0);
    }

    @Override
    public void setVariation(Enum variation) {
        this.m_variation = (VARIATION) variation;
    }

    @Override
    public String modelFamily() {
        return "keras_efficientnet";
    }

    @Override
    public String modelPrettyName() {
        switch (m_variation) {
            case EFFICIENTNET_B0:
                return "EfficientNetB0";
            case EFFICIENTNET_B1:
                return "EfficientNetB1";
            case EFFICIENTNET_B2:
                return "EfficientNetB2";
            case EFFICIENTNET_B3:
                return "EfficientNetB3";
            case EFFICIENTNET_B4:
                return "EfficientNetB4";
            case EFFICIENTNET_B5:
                return "EfficientNetB5";
            case EFFICIENTNET_B6:
                return "EfficientNetB6";
            case EFFICIENTNET_B7:
                return "EfficientNetB7";
            default:
                return null;
        }
    }
}
