package weka;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Main {
  public static void main(String[] args) {
    double[][] labelsD = {{1, 2, 3}, {0.1, 0.2, 0.3}, {6, 5, 4}};
    INDArray labels = Nd4j.create(labelsD);
    double[] means = {};
  }
}
