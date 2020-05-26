/*
 * WekaDeeplearning4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * WekaDeeplearning4j is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with WekaDeeplearning4j.  If not, see <https://www.gnu.org/licenses/>.
 *
 * ZooModelTest.java
 * Copyright (C) 2017-2018 University of Waikato, Hamilton, New Zealand
 */

package weka.zoo;

import java.io.File;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import javax.naming.OperationNotSupportedException;

import lombok.Data;
import lombok.extern.log4j.Log4j2;
import org.junit.BeforeClass;
import org.junit.Test;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.dl4j.PretrainedType;
import weka.dl4j.earlystopping.EarlyStopping;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.listener.EpochListener;
import weka.dl4j.zoo.*;
import weka.dl4j.zoo.Dl4jXception;
import weka.dl4j.zoo.keras.*;
import weka.dl4j.zoo.keras.NASNet;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Dl4jMlpFilter;
import weka.util.DatasetLoader;

import static org.junit.Assert.fail;

class ModelDownloader implements Runnable {

    AbstractZooModel zooModel;

    public ModelDownloader(AbstractZooModel zooModel) {
        this.zooModel = zooModel;
    }

    public void run() {
        this.zooModel.init(2, 0, new int[] {3, 224, 224}, true);
    }
}

/**
 * JUnit tests for the ModelZoo ({@link weka.zoo}). Mainly checks out whether the initialization of
 * the models work.
 *
 * @author Steven Lang
 */
@Log4j2
public class ZooModelTest {

    public static List<AbstractZooModel> createKerasModels() {
        List<AbstractZooModel> kerasModels = new ArrayList<>();

        KerasDenseNet kerasDenseNet121 = new KerasDenseNet();
        kerasDenseNet121.setVariation(DenseNet.VARIATION.DENSENET121);
        kerasModels.add(kerasDenseNet121);

        KerasDenseNet kerasDenseNet169 = new KerasDenseNet();
        kerasDenseNet169.setVariation(DenseNet.VARIATION.DENSENET169);
        kerasModels.add(kerasDenseNet169);

        KerasDenseNet kerasDenseNet201 = new KerasDenseNet();
        kerasDenseNet201.setVariation(DenseNet.VARIATION.DENSENET201);
        kerasModels.add(kerasDenseNet121);

//        EfficientNet efficientNetB0 = new EfficientNet();
//        efficientNetB0.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B0);
//        kerasModels.add(efficientNetB0);
//
//        EfficientNet efficientNetB1 = new EfficientNet();
//        efficientNetB1.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B1);
//        kerasModels.add(efficientNetB1);
//
//        EfficientNet efficientNetB2 = new EfficientNet();
//        efficientNetB2.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B2);
//        kerasModels.add(efficientNetB2);
//
//        EfficientNet efficientNetB3 = new EfficientNet();
//        efficientNetB3.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B3);
//        kerasModels.add(efficientNetB3);
//
//        EfficientNet efficientNetB4 = new EfficientNet();
//        efficientNetB4.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B4);
//        kerasModels.add(efficientNetB4);
//
//        EfficientNet efficientNetB5 = new EfficientNet();
//        efficientNetB5.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B5);
//        kerasModels.add(efficientNetB5);
//
//        EfficientNet efficientNetB6 = new EfficientNet();
//        efficientNetB6.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B6);
//        kerasModels.add(efficientNetB6);
//
//        EfficientNet efficientNetB7 = new EfficientNet();
//        efficientNetB7.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B7);
//        kerasModels.add(efficientNetB7);

//        InceptionResNetV2 inceptionResNetV2 = new InceptionResNetV2();
//        inceptionResNetV2.setVariation(InceptionResNetV2.VARIATION.STANDARD);
//        kerasModels.add(inceptionResNetV2);

        KerasInceptionV3 kerasInceptionV3 = new KerasInceptionV3();
        kerasModels.add(kerasInceptionV3);

//        MobileNet mobileNet = new MobileNet();
//        mobileNet.setVariation(MobileNet.VARIATION.V1);
//        kerasModels.add(mobileNet);
//
//        MobileNet mobileNetV2 = new MobileNet();
//        mobileNet.setVariation(MobileNet.VARIATION.V2);
//        kerasModels.add(mobileNetV2);

        KerasNASNet kerasNASNetMobile = new KerasNASNet();
        kerasNASNetMobile.setVariation(weka.dl4j.zoo.keras.NASNet.VARIATION.MOBILE);
        kerasModels.add(kerasNASNetMobile);

        KerasNASNet kerasNASNetLarge = new KerasNASNet();
        kerasNASNetLarge.setVariation(weka.dl4j.zoo.keras.NASNet.VARIATION.LARGE);
        kerasModels.add(kerasNASNetLarge);

        KerasResNet kerasResNet50 = new KerasResNet();
        kerasResNet50.setVariation(ResNet.VARIATION.RESNET50);
        kerasModels.add(kerasResNet50);

        KerasResNet kerasResNet50v2 = new KerasResNet();
        kerasResNet50v2.setVariation(ResNet.VARIATION.RESNET50V2);
        kerasModels.add(kerasResNet50v2);

        KerasResNet kerasResNet101 = new KerasResNet();
        kerasResNet101.setVariation(ResNet.VARIATION.RESNET101);
        kerasModels.add(kerasResNet101);

        KerasResNet kerasResNet101v2 = new KerasResNet();
        kerasResNet101v2.setVariation(ResNet.VARIATION.RESNET101V2);
        kerasModels.add(kerasResNet101v2);

        KerasResNet kerasResNet152 = new KerasResNet();
        kerasResNet152.setVariation(ResNet.VARIATION.RESNET152);
        kerasModels.add(kerasResNet152);

        KerasResNet kerasResNet152v2 = new KerasResNet();
        kerasResNet152v2.setVariation(ResNet.VARIATION.RESNET152V2);
        kerasModels.add(kerasResNet152v2);

        KerasVGG kerasVGG16 = new KerasVGG();
        kerasVGG16.setVariation(VGG.VARIATION.VGG16);
        kerasModels.add(kerasVGG16);

        KerasVGG kerasVGG19 = new KerasVGG();
        kerasVGG19.setVariation(VGG.VARIATION.VGG19);
        kerasModels.add(kerasVGG19);

        KerasXception kerasXception = new KerasXception();
        kerasModels.add(kerasXception);

        return kerasModels;
    }

    public static List<AbstractZooModel> createDL4JModels() {
        List<AbstractZooModel> dl4jModels = new ArrayList<>();

        Dl4jDarknet19 darknet19 = new Dl4jDarknet19();
        dl4jModels.add(darknet19);

        Dl4jLeNet leNet = new Dl4jLeNet();
        dl4jModels.add(leNet);

        Dl4JResNet50 resNet50 = new Dl4JResNet50();
        dl4jModels.add(resNet50);

        Dl4jSqueezeNet squeezeNet = new Dl4jSqueezeNet();
        dl4jModels.add(squeezeNet);

        Dl4jVGG vgg16 = new Dl4jVGG();
        dl4jModels.add(vgg16);

        Dl4jVGG vgg16VGGFace = new Dl4jVGG();
        vgg16VGGFace.setPretrainedType(PretrainedType.VGGFACE);
        dl4jModels.add(vgg16VGGFace);

        Dl4jVGG vgg19 = new Dl4jVGG();
        vgg19.setVariation(VGG.VARIATION.VGG19);
        dl4jModels.add(vgg19);

        Dl4jXception xception = new Dl4jXception();
        dl4jModels.add(xception);

        return dl4jModels;
    }

    @BeforeClass
    public static void downloadModels() {
        List<AbstractZooModel> dl4jModels = createDL4JModels();
        List<AbstractZooModel> kerasModels = createKerasModels();
        // Attempts to initialise pretrained versions of all models we're testing - via threads to speed up download
        ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() + 2);
        log.info("Ensuring zoo model weights are downloaded");
        for (AbstractZooModel zooModel : dl4jModels) {
            ModelDownloader modelDownloader = new ModelDownloader(zooModel);
            executor.execute(modelDownloader);
        }
        executor.shutdown();

        try {
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (Exception e) {
            e.printStackTrace();
        }
        log.info("Finished download zoo model weights");
    }

    // DL4J Model Tests
    @Test
    public void testAlexNetMnist() throws Exception {
        trainModel(new Dl4jAlexNet());
    }

    @Test
    public void testDarknet19() throws Exception {
        trainModel(new Dl4jDarknet19());
    }

    @Test
    public void testDarknet19Filter() throws Exception {
        filterModel(new Dl4jDarknet19());
    }

    @Test
    public void testFaceNetNN4Small2() throws Exception {
        trainModel(new Dl4jFaceNetNN4Small2());
    }

    @Test
    public void testInceptionResNetV1() throws Exception {
        trainModel(new Dl4jInceptionResNetV1());
    }

    @Test
    public void testLeNetMnist() throws Exception {
        trainModel(new Dl4jLeNet());
    }

    @Test
    public void testLeNetMnistFilter() throws Exception {
        filterModel(new Dl4jLeNet());
    }

//    @Test
//    public void testNASNet() throws Exception {
//      // NASNet has a bug in the initiation code https://github.com/eclipse/deeplearning4j/issues/7319
////      fail();
//        buildModel(new weka.dl4j.zoo.NASNet());
//    }

    @Test
    public void testResNet50() throws Exception {
        trainModel(new Dl4JResNet50());
    }

    @Test
    public void testResNet50Filter() throws Exception {
        filterModel(new Dl4JResNet50());
    }

    @Test
    public void testSqueezeNet() throws Exception {
        trainModel(new Dl4jSqueezeNet());
    }

    @Test
    public void testSqueezeNetFilter() throws Exception {
        filterModel(new Dl4jSqueezeNet());
    }

    @Test
    public void testVGG16() throws Exception {
        Dl4jVGG vgg16 = new Dl4jVGG();
        vgg16.setVariation(VGG.VARIATION.VGG16);
        trainModel(vgg16);
    }

    @Test
    public void testVGG16Filter() throws Exception {
        Dl4jVGG vgg16 = new Dl4jVGG();
        vgg16.setVariation(VGG.VARIATION.VGG16);
        filterModel(vgg16);
    }

    @Test
    public void testVGG19() throws Exception {
        Dl4jVGG vgg19 = new Dl4jVGG();
        vgg19.setVariation(VGG.VARIATION.VGG19);
        trainModel(vgg19);
    }

    @Test
    public void testVGG19Filter() throws Exception {
        Dl4jVGG vgg19 = new Dl4jVGG();
        vgg19.setVariation(VGG.VARIATION.VGG19);
        trainModel(vgg19);
    }

    @Test
    public void testXception() throws Exception {
        trainModel(new Dl4jXception());
    }

    @Test
    public void testXceptionFilter() throws Exception {
        filterModel(new Dl4jXception());
    }

    // Keras Zoo Models

    @Test
    public void testDenseNet121() throws Exception {
        KerasDenseNet kerasDenseNet = new KerasDenseNet();
        kerasDenseNet.setVariation(DenseNet.VARIATION.DENSENET121);
        trainModel(kerasDenseNet);
    }

    @Test
    public void testDenseNet121Filter() throws Exception {
        KerasDenseNet kerasDenseNet = new KerasDenseNet();
        kerasDenseNet.setVariation(DenseNet.VARIATION.DENSENET121);
        filterModel(kerasDenseNet);
    }

    @Test
    public void testDenseNet169() throws Exception {
        KerasDenseNet kerasDenseNet = new KerasDenseNet();
        kerasDenseNet.setVariation(DenseNet.VARIATION.DENSENET169);
        trainModel(kerasDenseNet);
    }

    @Test
    public void testDenseNet169Filter() throws Exception {
        KerasDenseNet kerasDenseNet = new KerasDenseNet();
        kerasDenseNet.setVariation(DenseNet.VARIATION.DENSENET169);
        filterModel(kerasDenseNet);
    }

    @Test
    public void testDenseNet201() throws Exception {
        KerasDenseNet kerasDenseNet = new KerasDenseNet();
        kerasDenseNet.setVariation(DenseNet.VARIATION.DENSENET201);
        trainModel(kerasDenseNet);
    }

    @Test
    public void testDenseNet201Filter() throws Exception {
        KerasDenseNet kerasDenseNet = new KerasDenseNet();
        kerasDenseNet.setVariation(DenseNet.VARIATION.DENSENET201);
        filterModel(kerasDenseNet);
    }

    /**
     * UNCOMMENT EFFICIENTNET TESTS WHEN ABLE TO RUN - waiting on new DL4J release
     */

//    @Test
//    public void testEfficientNetB0() throws Exception {
//        KerasEfficientNet kerasEfficientNet = new KerasEfficientNet();
//        kerasEfficientNet.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B0);
//        buildModel(kerasEfficientNet);
//    }
//
//    @Test
//    public void testEfficientNetB1() throws Exception {
//        KerasEfficientNet kerasEfficientNet = new KerasEfficientNet();
//        kerasEfficientNet.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B1);
//        buildModel(kerasEfficientNet);
//    }
//
//    @Test
//    public void testEfficientNetB2() throws Exception {
//        KerasEfficientNet kerasEfficientNet = new KerasEfficientNet();
//        kerasEfficientNet.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B2);
//        buildModel(kerasEfficientNet);
//    }
//
//    @Test
//    public void testEfficientNetB3() throws Exception {
//        KerasEfficientNet kerasEfficientNet = new KerasEfficientNet();
//        kerasEfficientNet.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B3);
//        buildModel(kerasEfficientNet);
//    }
//
//    @Test
//    public void testEfficientNetB4() throws Exception {
//        KerasEfficientNet kerasEfficientNet = new KerasEfficientNet();
//        kerasEfficientNet.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B4);
//        buildModel(kerasEfficientNet);
//    }
//
//    @Test
//    public void testEfficientNetB5() throws Exception {
//        KerasEfficientNet kerasEfficientNet = new KerasEfficientNet();
//        kerasEfficientNet.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B5);
//        buildModel(kerasEfficientNet);
//    }
//
//    @Test
//    public void testEfficientNetB6() throws Exception {
//        KerasEfficientNet kerasEfficientNet = new KerasEfficientNet();
//        kerasEfficientNet.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B6);
//        buildModel(kerasEfficientNet);
//    }
//
//    @Test
//    public void testEfficientNetB7() throws Exception {
//        KerasEfficientNet kerasEfficientNet = new KerasEfficientNet();
//        kerasEfficientNet.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B7);
//        buildModel(kerasEfficientNet);
//    }

    // InceptionResNetV2 does not work
//    @Test
//    public void testInceptionResNetV2() throws Exception {
//        KerasInceptionResNetV2 kerasInceptionResNetV2 = new KerasInceptionResNetV2();
//        kerasInceptionResNetV2.setVariation(InceptionResNetV2.VARIATION.STANDARD);
//        buildModel(kerasInceptionResNetV2);
//    }

    @Test
    public void testInceptionV3() throws Exception {
        KerasInceptionV3 kerasInceptionV3 = new KerasInceptionV3();
        kerasInceptionV3.setVariation(InceptionV3.VARIATION.STANDARD);
        trainModel(kerasInceptionV3);
    }

    @Test
    public void testInceptionV3Filter() throws Exception {
        KerasInceptionV3 kerasInceptionV3 = new KerasInceptionV3();
        kerasInceptionV3.setVariation(InceptionV3.VARIATION.STANDARD);
        filterModel(kerasInceptionV3);
    }

//    @Test
//    public void testMobileNetV1() throws Exception {
//        KerasMobileNet kerasMobileNet = new KerasMobileNet();
//        kerasMobileNet.setVariation(MobileNet.VARIATION.V1);
//        buildModel(kerasMobileNet);
//    }
//
//    @Test
//    public void testMobileNetV2() throws Exception {
//        KerasMobileNet kerasMobileNet = new KerasMobileNet();
//        kerasMobileNet.setVariation(MobileNet.VARIATION.V2);
//        buildModel(kerasMobileNet);
//    }

    @Test
    public void testNASNetMobile() throws Exception {
        KerasNASNet kerasNASNet = new KerasNASNet();
        kerasNASNet.setVariation(NASNet.VARIATION.MOBILE);
        trainModel(kerasNASNet);
    }

    @Test
    public void testNASNetMobileFilter() throws Exception {
        KerasNASNet kerasNASNet = new KerasNASNet();
        kerasNASNet.setVariation(NASNet.VARIATION.MOBILE);
        filterModel(kerasNASNet);
    }

    @Test
    public void testNASNetLarge() throws Exception {
        KerasNASNet kerasNASNet = new KerasNASNet();
        kerasNASNet.setVariation(NASNet.VARIATION.LARGE);
        trainModel(kerasNASNet);
    }

    @Test
    public void testNASNetLargeFilter() throws Exception {
        KerasNASNet kerasNASNet = new KerasNASNet();
        kerasNASNet.setVariation(NASNet.VARIATION.LARGE);
        filterModel(kerasNASNet);
    }

    @Test
    public void testKerasResnet50() throws Exception {
        KerasResNet kerasResNet = new KerasResNet();
        kerasResNet.setVariation(ResNet.VARIATION.RESNET50);
        trainModel(kerasResNet);
    }

    @Test
    public void testKerasResnet50Filter() throws Exception {
        KerasResNet kerasResNet = new KerasResNet();
        kerasResNet.setVariation(ResNet.VARIATION.RESNET50);
        filterModel(kerasResNet);
    }

    @Test
    public void testKerasResnet50V2() throws Exception {
        KerasResNet kerasResNet = new KerasResNet();
        kerasResNet.setVariation(ResNet.VARIATION.RESNET50V2);
        trainModel(kerasResNet);
    }

    @Test
    public void testKerasResnet50V2Filter() throws Exception {
        KerasResNet kerasResNet = new KerasResNet();
        kerasResNet.setVariation(ResNet.VARIATION.RESNET50V2);
        filterModel(kerasResNet);
    }

    @Test
    public void testKerasResnet101() throws Exception {
        KerasResNet kerasResNet = new KerasResNet();
        kerasResNet.setVariation(ResNet.VARIATION.RESNET101);
        trainModel(kerasResNet);
    }

    @Test
    public void testKerasResnet101Filter() throws Exception {
        KerasResNet kerasResNet = new KerasResNet();
        kerasResNet.setVariation(ResNet.VARIATION.RESNET101);
        filterModel(kerasResNet);
    }

    @Test
    public void testKerasResnet101V2() throws Exception {
        KerasResNet kerasResNet = new KerasResNet();
        kerasResNet.setVariation(ResNet.VARIATION.RESNET101V2);
        trainModel(kerasResNet);
    }

    @Test
    public void testKerasResnet101V2Filter() throws Exception {
        KerasResNet kerasResNet = new KerasResNet();
        kerasResNet.setVariation(ResNet.VARIATION.RESNET101V2);
        filterModel(kerasResNet);
    }

    @Test
    public void testKerasResnet152() throws Exception {
        KerasResNet kerasResNet = new KerasResNet();
        kerasResNet.setVariation(ResNet.VARIATION.RESNET152);
        trainModel(kerasResNet);
    }

    @Test
    public void testKerasResnet152Filter() throws Exception {
        KerasResNet kerasResNet = new KerasResNet();
        kerasResNet.setVariation(ResNet.VARIATION.RESNET152);
        filterModel(kerasResNet);
    }

    @Test
    public void testKerasResnet152V2() throws Exception {
        KerasResNet kerasResNet = new KerasResNet();
        kerasResNet.setVariation(ResNet.VARIATION.RESNET152V2);
        trainModel(kerasResNet);
    }

    @Test
    public void testKerasResnet152V2Filter() throws Exception {
        KerasResNet kerasResNet = new KerasResNet();
        kerasResNet.setVariation(ResNet.VARIATION.RESNET152V2);
        filterModel(kerasResNet);
    }

    @Test
    public void testKerasVGG16() throws Exception {
        KerasVGG kerasVGG = new KerasVGG();
        kerasVGG.setVariation(VGG.VARIATION.VGG16);
        trainModel(kerasVGG);
    }

    @Test
    public void testKerasVGG16Filter() throws Exception {
        KerasVGG kerasVGG = new KerasVGG();
        kerasVGG.setVariation(VGG.VARIATION.VGG16);
        filterModel(kerasVGG);
    }

    @Test
    public void testKerasVGG19() throws Exception {
        KerasVGG kerasVGG = new KerasVGG();
        kerasVGG.setVariation(VGG.VARIATION.VGG19);
        trainModel(kerasVGG);
    }

    @Test
    public void testKerasVGG19Filter() throws Exception {
        KerasVGG kerasVGG = new KerasVGG();
        kerasVGG.setVariation(VGG.VARIATION.VGG19);
        filterModel(kerasVGG);
    }

    @Test
    public void testKerasXception() throws Exception {
        KerasXception kerasXception = new KerasXception();
        kerasXception.setVariation(weka.dl4j.zoo.keras.Xception.VARIATION.STANDARD);
        trainModel(kerasXception);
    }

    @Test
    public void testKerasXceptionFilter() throws Exception {
        KerasXception kerasXception = new KerasXception();
        kerasXception.setVariation(weka.dl4j.zoo.keras.Xception.VARIATION.STANDARD);
        filterModel(kerasXception);
    }

    private void filterModel(AbstractZooModel model) throws Exception {
        Dl4jMlpFilter myFilter = new Dl4jMlpFilter();
        ImageInstanceIterator iterator = DatasetLoader.loadMiniMnistImageIterator();
        Instances shrunkenInstances = shrinkInstances(DatasetLoader.loadMiniMnistMeta());
        myFilter.setZooModelType(model);
        myFilter.setImageInstanceIterator(iterator);
        myFilter.setInputFormat(shrunkenInstances);
        Filter.useFilter(shrunkenInstances, myFilter);
    }

    private Instances shrinkInstances(Instances data) {
        ArrayList<Attribute> atts = new ArrayList<>();
        for (int i = 0; i < data.numAttributes(); i++) {
            atts.add(data.attribute(i));
        }
        Instances shrunkenData = new Instances("shrinked", atts, 10);
        shrunkenData.setClassIndex(1);
        for (int i = 0; i < 10; i++) {
            Instance inst = data.get(i);
            inst.setClassValue(i % 10);
            inst.setDataset(shrunkenData);
            shrunkenData.add(inst);
        }
        return shrunkenData;
    }

    private void trainModel(AbstractZooModel model) throws Exception {
        // CLF
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);

        // Data
        Instances data = DatasetLoader.loadMiniMnistMeta();
        Instances shrunkenData = shrinkInstances(data);

        ImageInstanceIterator iterator = DatasetLoader.loadMiniMnistImageIterator();
        iterator.setTrainBatchSize(10);
        clf.setInstanceIterator(iterator);
        clf.setZooModel(model);
        clf.setNumEpochs(1);
        final EpochListener epochListener = new EpochListener();
        epochListener.setN(1);
        clf.setIterationListener(epochListener);
        clf.setEarlyStopping(new EarlyStopping(5, 0));
        clf.buildClassifier(shrunkenData);
    }


    /**
     * Test CustomNet init
     */
    @Test(expected = UnsupportedOperationException.class)
    public void testCustomNetInit() throws OperationNotSupportedException {
        new CustomNet().init(0, 0, null, false);
    }
}
