package weka.core.progress;

// TODO document

public abstract class AbstractProgressBar {

    protected boolean m_indeterminate = false; // TODO implement

    protected double m_actualProgress = 0;

    // value between 0 and 1
    protected double m_normalizedProgress = 0;

    protected double m_maxProgress = 0;

    public void reset(double maxProgress) {
        m_normalizedProgress = 0;
        m_actualProgress = 0;
        setMax(maxProgress);
    }

    public void setProgress(double progress) {
        m_actualProgress = progress;
        m_normalizedProgress = m_actualProgress / m_maxProgress;
    }

    public void increment() {
        setProgress(m_actualProgress + 1);
    }

    public abstract void show();

    public abstract void hide();

    public abstract void refreshDisplay();

    public double getMax() {
        return m_maxProgress;
    }

    public void setMax(double max) {
        this.m_maxProgress = max;
    }
}
