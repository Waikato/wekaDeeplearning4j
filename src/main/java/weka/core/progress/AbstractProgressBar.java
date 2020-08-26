package weka.core.progress;

import java.io.Serializable;
import java.util.concurrent.TimeUnit;

// TODO document
// TODO calculate ETA
public abstract class AbstractProgressBar implements Serializable {

    protected boolean m_indeterminate = false; // TODO implement

    protected String m_progressMessage;

    protected double m_actualProgress = 0;

    // value between 0 and 1
    protected double m_normalizedProgress = 0;

    protected double m_maxProgress = 0;

    protected String etaHms = "";

    protected long startTime;

    public AbstractProgressBar() { }

    public AbstractProgressBar(double maxProgress, String progressMessage) {
        setMaxProgress(maxProgress);
        setProgressMessage(progressMessage);
    }

    public void start() {
        m_normalizedProgress = 0;
        m_actualProgress = 0;
        startTime = System.currentTimeMillis();
        onStart();
        refreshDisplay();
    }

    protected abstract void onStart();

    protected abstract void onSetProgress();

    public void increment() {
        setProgress(m_actualProgress + 1);
    }

    public abstract void finish();

    public abstract void refreshDisplay();

    public abstract void show();

    public abstract void finish();

    public void setProgress(double progress) {
        // Limit the progress to the max previously set
        m_actualProgress = Math.min(progress, m_maxProgress);

        calculate();

        onSetProgress();
        refreshDisplay();
    }

    public double getMaxProgress() {
        return m_maxProgress;
    }

    public void setMaxProgress(double max) {
        if (max < 0) {
            m_indeterminate = true;
        }
        this.m_maxProgress = max;
    }

    public String getProgressMessage() {
        return m_progressMessage;
    }

    public void setProgressMessage(String message) {
        m_progressMessage = message;
    }

    public String getETAString() {
        return String.format("ETA: %s", etaHms);
    }
}
