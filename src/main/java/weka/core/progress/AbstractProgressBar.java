package weka.core.progress;

// TODO document
// TODO calculate ETA
public abstract class AbstractProgressBar {

    public AbstractProgressBar() { }

    public AbstractProgressBar(double maxProgress, String progressMessage) {
        reset();
        setMax(maxProgress);
        m_progressMessage = progressMessage;
    }

    protected boolean m_indeterminate = false; // TODO implement

    protected String m_progressMessage;

    protected double m_actualProgress = 0;

    // value between 0 and 1
    protected double m_normalizedProgress = 0;

    protected double m_maxProgress = 0;

    public void reset() {
        m_normalizedProgress = 0;
        m_actualProgress = 0;
    }

    public void setProgress(double progress) {
        m_actualProgress = progress;
        m_normalizedProgress = m_actualProgress / m_maxProgress;
        onSetProgress();
        refreshDisplay();
    }

    public void increment() {
        setProgress(m_actualProgress + 1);
    }

    protected abstract void onSetProgress();

    public abstract void show();

    public abstract void finish();

    public abstract void refreshDisplay();

    public double getMax() {
        return m_maxProgress;
    }

    public void setMax(double max) {
        if (max < 0) {
            m_indeterminate = true;
        }
        this.m_maxProgress = max;
    }
}
