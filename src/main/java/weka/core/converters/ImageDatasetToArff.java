package weka.core.converters;

import weka.core.*;

import java.io.File;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class ImageDatasetToArff {

    public String[] fileListForFolder(File folder) {
        String[] imageNames = folder.list();
        String imgClass = folder.getName();
        for (int i = 0; i < imageNames.length; i++) {
            String imgName = imageNames[i];
            if (!imgName.toLowerCase().endsWith(".jpg") && !imgName.toLowerCase().endsWith(".png")) {
                System.err.println(String.format("Found non image file: %s, ignoring...", imgName));
                continue;
            }
            imageNames[i] = Paths.get(imgClass, imgName).toString();
        }
        return imageNames;
    }

    public Instances createDataset(String directoryPath) throws Exception {

        File dir = new File(directoryPath);
        File[] classArr = dir.listFiles();
        assert classArr != null;
        List<File> classes = Arrays.stream(classArr).filter(File::isDirectory).collect(Collectors.toList());
        List<String> classStrings = classes.stream().map(File::getName).collect(Collectors.toList());

        ArrayList<Attribute> atts = new ArrayList<>(2);
        atts.add(new Attribute("image_path", (List<String>) null));
        atts.add(new Attribute("class", classStrings));

        Instances data = new Instances("image_dataset__" + dir.getName(), atts, 0);

        for (File imgClass : classes) {
            String[] imgs = fileListForFolder(imgClass);
            for (String img : imgs) {
                double[] newInst = new double[2];
                newInst[0] = data.attribute(0).addStringValue(img);
                newInst[1] = classStrings.indexOf(imgClass.getName());
                data.add(new DenseInstance(1.0, newInst));
            }
        }
        return data;
    }

    public static void main(String[] args) {
        if (args.length == 1) {
            ImageDatasetToArff imageDatasetToArff = new ImageDatasetToArff();
            try {
                Instances dataset = imageDatasetToArff.createDataset(args[0]);
                System.out.println(dataset);
            } catch (Exception e) {
                System.err.println(e.getMessage());
                e.printStackTrace();
            }
        } else {
            System.out.println("Usage: java ImageDatasetToArff <directory name>");
        }
    }
}
