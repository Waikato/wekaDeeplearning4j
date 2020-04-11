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

import java.io.IOException;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import javax.naming.OperationNotSupportedException;

import lombok.extern.log4j.Log4j2;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.zoo.ZooModel;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.shade.protobuf.MapEntry;
import weka.classifiers.functions.Dl4jMlpClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.dl4j.Preferences;
import weka.dl4j.PretrainedType;
import weka.dl4j.earlystopping.EarlyStopping;
import weka.dl4j.iterators.instance.ImageInstanceIterator;
import weka.dl4j.listener.EpochListener;
import weka.dl4j.zoo.*;
import weka.dl4j.zoo.NASNet;
import weka.dl4j.zoo.Xception;
import weka.dl4j.zoo.keras.*;
import weka.util.DatasetLoader;

import static org.junit.Assert.fail;

class ModelDownloader implements Runnable {

    ZooModel zooModel;
    PretrainedType pretrainedType;

    public ModelDownloader(ZooModel zooModel, PretrainedType pretrainedType) {
        this.zooModel = zooModel;
        this.pretrainedType = pretrainedType;
    }

    public void run() {
        try {
            this.zooModel.initPretrained(this.pretrainedType.getBackend());
        } catch (IOException ex) {
            ex.printStackTrace();
        }
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

    private static Map<ZooModel, PretrainedType> modelsToDownload;

    private static List<ZooModel> createKerasModelVariations() {
        List<ZooModel> kerasModels = new ArrayList<>();

        DenseNet denseNet121 = new DenseNet();
        denseNet121.setVariation(DenseNet.VARIATION.DENSENET121);
        kerasModels.add(denseNet121);

        DenseNet denseNet169 = new DenseNet();
        denseNet169.setVariation(DenseNet.VARIATION.DENSENET169);
        kerasModels.add(denseNet169);

        DenseNet denseNet201 = new DenseNet();
        denseNet201.setVariation(DenseNet.VARIATION.DENSENET201);
        kerasModels.add(denseNet201);

        EfficientNet efficientNetB0 = new EfficientNet();
        efficientNetB0.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B0);
        kerasModels.add(efficientNetB0);

        EfficientNet efficientNetB1 = new EfficientNet();
        efficientNetB1.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B1);
        kerasModels.add(efficientNetB1);

        EfficientNet efficientNetB2 = new EfficientNet();
        efficientNetB2.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B2);
        kerasModels.add(efficientNetB2);

        EfficientNet efficientNetB3 = new EfficientNet();
        efficientNetB3.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B3);
        kerasModels.add(efficientNetB3);

        EfficientNet efficientNetB4 = new EfficientNet();
        efficientNetB4.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B4);
        kerasModels.add(efficientNetB4);

        EfficientNet efficientNetB5 = new EfficientNet();
        efficientNetB5.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B5);
        kerasModels.add(efficientNetB5);

        EfficientNet efficientNetB6 = new EfficientNet();
        efficientNetB6.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B6);
        kerasModels.add(efficientNetB6);

        EfficientNet efficientNetB7 = new EfficientNet();
        efficientNetB7.setVariation(EfficientNet.VARIATION.EFFICIENTNET_B7);
        kerasModels.add(efficientNetB7);

        InceptionResNetV2 inceptionResNetV2 = new InceptionResNetV2();
        inceptionResNetV2.setVariation(InceptionResNetV2.VARIATION.STANDARD);
        kerasModels.add(inceptionResNetV2);

        InceptionV3 inceptionV3 = new InceptionV3();
        inceptionV3.setVariation(InceptionV3.VARIATION.STANDARD);
        kerasModels.add(inceptionV3);

        MobileNet mobileNet = new MobileNet();
        mobileNet.setVariation(MobileNet.VARIATION.V1);
        kerasModels.add(mobileNet);

        MobileNet mobileNetV2 = new MobileNet();
        mobileNet.setVariation(MobileNet.VARIATION.V2);
        kerasModels.add(mobileNetV2);

        weka.dl4j.zoo.keras.NASNet nasNetMobile = new weka.dl4j.zoo.keras.NASNet();
        nasNetMobile.setVariation(weka.dl4j.zoo.keras.NASNet.VARIATION.MOBILE);
        kerasModels.add(nasNetMobile);

        weka.dl4j.zoo.keras.NASNet nasNetLarge = new weka.dl4j.zoo.keras.NASNet();
        nasNetLarge.setVariation(weka.dl4j.zoo.keras.NASNet.VARIATION.LARGE);
        kerasModels.add(nasNetLarge);

        ResNet resNet50 = new ResNet();
        resNet50.setVariation(ResNet.VARIATION.RESNET50);
        kerasModels.add(resNet50);

        ResNet resNet50V2 = new ResNet();
        resNet50V2.setVariation(ResNet.VARIATION.RESNET50V2);
        kerasModels.add(resNet50V2);

        ResNet resNet101 = new ResNet();
        resNet101.setVariation(ResNet.VARIATION.RESNET101);
        kerasModels.add(resNet101);

        ResNet resNet101V2 = new ResNet();
        resNet101V2.setVariation(ResNet.VARIATION.RESNET101V2);
        kerasModels.add(resNet101V2);

        ResNet resNet152 = new ResNet();
        resNet152.setVariation(ResNet.VARIATION.RESNET152);
        kerasModels.add(resNet152);

        ResNet resNet152V2 = new ResNet();
        resNet152V2.setVariation(ResNet.VARIATION.RESNET152V2);
        kerasModels.add(resNet152V2);

        VGG vgg16 = new VGG();
        vgg16.setVariation(VGG.VARIATION.VGG16);
        kerasModels.add(vgg16);

        VGG vgg19 = new VGG();
        vgg19.setVariation(VGG.VARIATION.VGG19);
        kerasModels.add(vgg19);

        weka.dl4j.zoo.keras.Xception xception = new weka.dl4j.zoo.keras.Xception();
        xception.setVariation(weka.dl4j.zoo.keras.Xception.VARIATION.STANDARD);
        kerasModels.add(xception);

        return kerasModels;
    }

    private static void createModelsToDownload() {
        Map<ZooModel, PretrainedType> models = new HashMap<>();
        models.put(org.deeplearning4j.zoo.model.Darknet19.builder().build(), PretrainedType.IMAGENET);
        models.put(org.deeplearning4j.zoo.model.LeNet.builder().build(), PretrainedType.MNIST);
        models.put(org.deeplearning4j.zoo.model.ResNet50.builder().build(), PretrainedType.IMAGENET);
        models.put(org.deeplearning4j.zoo.model.SqueezeNet.builder().build(), PretrainedType.IMAGENET);
        models.put(org.deeplearning4j.zoo.model.VGG16.builder().build(), PretrainedType.IMAGENET);
        models.put(org.deeplearning4j.zoo.model.VGG19.builder().build(), PretrainedType.IMAGENET);
        models.put(org.deeplearning4j.zoo.model.Xception.builder().build(), PretrainedType.IMAGENET);

        List<ZooModel> kerasModels = createKerasModelVariations();
        for (ZooModel zooModel : kerasModels) {
          models.put(zooModel, PretrainedType.IMAGENET);
        }

        modelsToDownload = Collections.unmodifiableMap(models);
    }

    @BeforeClass
    public static void downloadModels() {
        createModelsToDownload();
        // Attempts to initialise pretrained versions of all models we're testing - via threads to speed up download
        ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() + 2);
        log.info("Ensuring zoo model weights are downloaded");
        for (Map.Entry<ZooModel, PretrainedType> mapEntry : modelsToDownload.entrySet()) {
            ModelDownloader modelDownloader = new ModelDownloader(mapEntry.getKey(), mapEntry.getValue());
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

    @Test
    public void testAlexNetMnist() throws Exception {
        buildModel(new AlexNet());
    }

    @Test
    public void testDarknet19() throws Exception {
        buildModel(new Darknet19());
    }

    @Test
    public void testFaceNetNN4Small2() throws Exception {
        buildModel(new FaceNetNN4Small2());
    }

    @Test
    public void testInceptionResNetV1() throws Exception {
        buildModel(new InceptionResNetV1());
    }

    @Test
    public void testLeNetMnist() throws Exception {
        buildModel(new LeNet());
    }

    @Test
    public void testNASNet() throws Exception {
      // NASNet has a bug in the initiation code
      fail();
//        buildModel(new NASNet());
    }

    @Test
    public void testResNet50() throws Exception {
        buildModel(new ResNet50());
    }

    @Test
    public void testSqueezeNet() throws Exception {
        buildModel(new SqueezeNet());
    }

    @Test
    public void testVGG16() throws Exception {
        buildModel(new VGG16());
    }

    @Test
    public void testVGG19() throws Exception {
        buildModel(new VGG19());
    }

    @Test
    public void testXception() throws Exception {
        buildModel(new Xception());
    }


    private void buildModel(AbstractZooModel model) throws Exception {
        // CLF
        Dl4jMlpClassifier clf = new Dl4jMlpClassifier();
        clf.setSeed(1);

        // Data
        Instances data = DatasetLoader.loadMiniMnistMeta();

        ArrayList<Attribute> atts = new ArrayList<>();
        for (int i = 0; i < data.numAttributes(); i++) {
            atts.add(data.attribute(i));
        }
        Instances shrinkedData = new Instances("shrinked", atts, 10);
        shrinkedData.setClassIndex(1);
        for (int i = 0; i < 10; i++) {
            Instance inst = data.get(i);
            inst.setClassValue(i % 10);
            inst.setDataset(shrinkedData);
            shrinkedData.add(inst);
        }

        ImageInstanceIterator iterator = DatasetLoader.loadMiniMnistImageIterator();
        iterator.setTrainBatchSize(10);
        clf.setInstanceIterator(iterator);
        clf.setZooModel(model);
        clf.setNumEpochs(1);
        final EpochListener epochListener = new EpochListener();
        epochListener.setN(1);
        clf.setIterationListener(epochListener);
        clf.setEarlyStopping(new EarlyStopping(5, 0));
        clf.buildClassifier(shrinkedData);
    }


    /**
     * Test CustomNet init
     */
    @Test(expected = UnsupportedOperationException.class)
    public void testCustomNetInit() throws OperationNotSupportedException {
        new CustomNet().init(0, 0, null, false);
    }
}
