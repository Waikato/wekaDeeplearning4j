package weka.dl4j.playground;

public class TopNPredictions {

    protected int n = 5;

    protected Prediction[] topNPredictions;

    public TopNPredictions() {
        initPredArray();
    }

    public TopNPredictions(int n) {
        setN(n);
        initPredArray();
    }

    public void addPrediction(int i, Prediction prediction) {
        this.topNPredictions[i] = prediction;
    }

    protected void initPredArray() {
        this.topNPredictions = new Prediction[n];
    }

    public void setN(int n) {
        this.n = n;
    }

    public String toSummaryString() {
        StringBuffer text = new StringBuffer();

    }
}
