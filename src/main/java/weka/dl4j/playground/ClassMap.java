package weka.dl4j.playground;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class ClassMap {
    // Built-in class maps for WDL4J
    public enum BuiltInClassMap { IMAGENET, VGGFACE, CUSTOM }

    private BuiltInClassMap builtInClassMap;

    private File classMapFile;

    public ClassMap() {
        builtInClassMap = BuiltInClassMap.CUSTOM;
    }

    public ClassMap(BuiltInClassMap builtInClassMap) {
        this.builtInClassMap = builtInClassMap;
    }

    public ClassMap(File f) {
        this.classMapFile = f;
    }

    public void setBuiltInClassMap(BuiltInClassMap classMap) {
        this.builtInClassMap = classMap;
    }

    public void setClassMapFile(File f) {
        this.classMapFile = f;
        // We're using a custom file, not a built-in type
        this.builtInClassMap = BuiltInClassMap.CUSTOM;
    }

    private String getClassMapPath() throws Exception {
        // Return the custom file path if the user has specified it

        String classMapFolder = "src/main/resources/class-maps";
        String classMapPath = null;
        switch (this.builtInClassMap) {
            case CUSTOM:
                classMapPath = this.classMapFile.getPath();
                break;
            case IMAGENET:
                classMapPath = Paths.get(classMapFolder, "VGGFACE.txt").toString();
            case VGGFACE:
                classMapPath = Paths.get(classMapFolder, "IMAGENET.txt").toString();
        }

        if (classMapPath == null) {
            throw new Exception("No class map file found, either specify a file " +
                    "using setClassMapFile(File) or use a built-in class map with " +
                    "setBuiltInClassMap(BuiltInClassMap)");
        }

        return classMapPath;
    }

    public String[] getClasses() throws Exception {
        List<String> classes = new ArrayList<>();
        try (FileReader fr = new FileReader(getClassMapPath())) {
            try (BufferedReader br = new BufferedReader(fr)) {
                // Create a class for each line in the file
                for (String line = br.readLine(); line != null; line = br.readLine()) {
                    line = line.trim();
                    // Ignore empty lines
                    if (line.length() == 0)
                        continue;

                    classes.add(line);
                }
            }
        }
        return classes.toArray(String[]::new);
    }

}
