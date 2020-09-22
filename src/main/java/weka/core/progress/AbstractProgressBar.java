package weka.core.progress;

import java.io.Serializable;
import java.util.concurrent.TimeUnit;

// TODO document
public abstract class AbstractProgressBar implements Serializable {

    protected boolean m_indeterminate = false;

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
        calculate();
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

    private void calculate() {
        m_normalizedProgress = m_actualProgress / m_maxProgress;

        if (m_indeterminate) {
            return;
        }

        long eta = (long) ((m_maxProgress - m_actualProgress) * (System.currentTimeMillis() - startTime) / m_actualProgress);

        etaHms = String.format("%02d:%02d:%02d", TimeUnit.MILLISECONDS.toHours(eta),
                TimeUnit.MILLISECONDS.toMinutes(eta) % TimeUnit.HOURS.toMinutes(1),
                TimeUnit.MILLISECONDS.toSeconds(eta) % TimeUnit.MINUTES.toSeconds(1));
    }

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
        m_maxProgress = max;
        m_indeterminate = max < 0;
    }

    public String getProgressMessage() {
        return m_progressMessage;
    }

    public void setProgressMessage(String message) {
        m_progressMessage = message;
    }

    public String getETAString() {
        return m_indeterminate ? "" : String.format("ETA: %s", etaHms);
    }
}
