package weka.dl4j.playground;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

public class TopNPredictions {

    protected int n = 5;

    protected Prediction[] topNPredictions;

    public TopNPredictions() { }

    public TopNPredictions(int n) {
        setN(n);
    }

    protected void initPredArray() {
        this.topNPredictions = new Prediction[n];
    }

    public void setN(int n) {
        this.n = n;
    }

    public Prediction getPrediction(int n) {
        return topNPredictions[n];
    }

    public Prediction getTopPrediction() {
        return topNPredictions[0];
    }

    public void process(INDArray predictions, ClassMap classMap) throws Exception {
        initPredArray();
        String[] classes = classMap.getClasses();

        double[] predDouble = predictions.toDoubleVector();
        int[] bestNPredictions = getBestKIndices(predDouble, n);

        for (int i = 0; i < bestNPredictions.length; i++) {
            int topNIndex = bestNPredictions[i];
            String className = classes[topNIndex];
            double classProb = predictions.getDouble(topNIndex);

            Prediction p = new Prediction(topNIndex, className, classProb);

            this.topNPredictions[i] = p;
        }
    }

    private int getMaxLenClassName() {
        // Default length is 15
        int maxLen = 15;
        for (Prediction p : topNPredictions) {
            int tmpLen = p.getClassName().length();
            if (tmpLen > maxLen) {
                maxLen = tmpLen;
            }
        }
        return maxLen;
    }

    private String getTableBreak(String breakChar, int len) {
        return breakChar.repeat(len) + "\n";
    }

    public String toSummaryString(String imageName, String zooModelName) {
        StringBuilder text = new StringBuilder();

        int maxLenClassName = getMaxLenClassName();

        String titleClassID = "Class ID";
        String titleClassName = "Class Name";
        String titleProbability = "Prob %";

        // The class name column should be based on the maximum length of the top 5 class names
        String formattedClassName = "%" + maxLenClassName + "s"; // --> " %20s "

        // Format used for each line in the table, keeps all columns equal
        String lineFormat = "%10s | " + formattedClassName + " | %10s\n";

        // -> "Class ID |               Class Name |     Prob %"
        String columnHeaders = String.format(lineFormat, titleClassID, titleClassName, titleProbability);

        int lineLength = columnHeaders.length();

        // Add the top table break
        text.append(getTableBreak("=", lineLength));

        // Add the image name and the model used for prediction
        text.append(imageName).append(" - ").append(zooModelName).append("\n\n");

        // Generate the column headers
        text.append(columnHeaders);

        text.append(getTableBreak("-", lineLength));

        // Add each row to the output table
        for (Prediction p : topNPredictions) {
            String classID = "" + p.getClassID();
            String probability = String.format("%.3f", p.getClassProbability());
            text.append(String.format(lineFormat, classID, p.getClassName(), probability));
        }

        text.append(getTableBreak("=", lineLength));

        return text.toString();
    }

    private int[] getBestKIndices(double[] array, int num) {
        //create sort able array with index and value pair
        IndexValuePair[] pairs = new IndexValuePair[array.length];
        for (int i = 0; i < array.length; i++) {
            pairs[i] = new IndexValuePair(i, array[i]);
        }

        //sort
        Arrays.sort(pairs, (o1, o2) -> Double.compare(o2.value, o1.value));

        //extract the indices
        int[] result = new int[num];
        for (int i = 0; i < num; i++) {
            result[i] = pairs[i].index;
        }
        return result;
    }

    private static class IndexValuePair {
        private final int index;
        private final double value;

        public IndexValuePair(int index, double value) {
            this.index = index;
            this.value = value;
        }
    }
}
