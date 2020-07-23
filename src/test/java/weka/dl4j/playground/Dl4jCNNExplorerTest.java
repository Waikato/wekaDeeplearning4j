package weka.dl4j.playground;

import junit.framework.TestCase;
import weka.dl4j.PretrainedType;
import weka.dl4j.zoo.Dl4jVGG;
import weka.dl4j.zoo.KerasResNet;
import weka.dl4j.zoo.keras.VGG;

import java.io.File;

public class Dl4jCNNExplorerTest extends TestCase {

    private final static int BEN_STILLER_ID = 201;

    private final static int TROMBONE_ID = 875;

    /**
     * Test the explorer with VGG16 pretrained on VGGFACE, and a photo of Ben Stiller
     * @throws Exception
     */
    public void testBenStillerVGGFACE() throws Exception {
        Dl4jCNNExplorer explorer = new Dl4jCNNExplorer();

        Dl4jVGG zooModel = new Dl4jVGG();
        zooModel.setVariation(VGG.VARIATION.VGG16);
        zooModel.setPretrainedType(PretrainedType.VGGFACE);
        explorer.setZooModelType(zooModel);

        ModelOutputDecoder decoder = new ModelOutputDecoder();
        decoder.setBuiltInClassMap(ModelOutputDecoder.ClassmapType.VGGFACE);
        explorer.setModelOutputDecoder(decoder);

        checkExplorerClassID(explorer, "src/test/resources/images/ben_stiller.jpg", BEN_STILLER_ID);
    }

    /**
     * Test the explorer with a pretrained ResNet on ImageNet and a photo of a trombone
     * @throws Exception
     */
    public void testTromboneResNet() throws Exception {
        Dl4jCNNExplorer explorer = new Dl4jCNNExplorer();

        KerasResNet zooModel = new KerasResNet();
        explorer.setZooModelType(zooModel);

        checkExplorerClassID(explorer, "src/test/resources/images/trombone.jpg", TROMBONE_ID);
    }

    private void checkExplorerClassID(Dl4jCNNExplorer explorer, String imagePath, int expectedClassID) throws Exception {
        explorer.init();
        explorer.makePrediction(new File(imagePath));

        TopNPredictions topNPredictions = explorer.getCurrentPredictions();

        assertEquals(expectedClassID, topNPredictions.getPrediction(0).getClassID());
    }
}