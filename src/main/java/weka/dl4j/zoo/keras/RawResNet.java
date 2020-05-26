package weka.dl4j.zoo.keras;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;

public class RawResNet extends ZooModel {
    private static final long serialVersionUID = 53159273073018L;

    @Override
    public void setInputShape(int[][] inputShape) {
        this.inputShape = inputShape[0];
    }

    @Override
    public ModelMetaData metaData() {
        return null;
    }

    @Override
    public Class<? extends Model> modelType() {
        return ComputationGraph.class;
    }

    @Override
    public String pretrainedUrl(PretrainedType pretrainedType) {
        return "https://github.com/basedrhys/wekaDeeplearning4j/releases/download/zoo-models-v2/KerasResNet50.zip";
    }

    @Override
    public long pretrainedChecksum(PretrainedType pretrainedType) {
        return 3286468754L;
    }

    @Override
    public ComputationGraph init() {
        return null;
    }

    public enum VARIATION {RESNET50, RESNET50V2, RESNET101, RESNET101V2, RESNET152, RESNET152V2};

    public static int[] inputShape = new int[] {3, 224, 224};

    protected VARIATION m_variation = VARIATION.RESNET50;

    public RawResNet() {
        setVariation(VARIATION.RESNET50);
    }

//    @Override
    public void setVariation(Enum variation) {
        this.m_variation = (VARIATION) variation;
    }

//    @Override
    public String modelFamily() {
        return "keras_resnet";
    }

//    @Override
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
