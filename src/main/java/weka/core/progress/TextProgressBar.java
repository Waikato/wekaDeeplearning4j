package weka.core.progress;

import lombok.extern.log4j.Log4j2;

// TODO document
// TODO implement indeterminate bar

@Log4j2
public class TextProgressBar extends AbstractProgressBar {

    public TextProgressBar(double maxProgress, String progressMessage) {
        super(maxProgress, progressMessage);
    }

    private String progressChar = "=";

    private String progressRemainingChar = " ";

    protected int progressBarSize = 60;

    private int numDots = 0;

    public void setProgress(double progress) {
        super.setProgress(progress);
        numDots = (int) (m_normalizedProgress * progressBarSize);

        refreshDisplay();
    }

    public void refreshDisplay() {
        System.err.print(String.format("\r%s: [%s%s]", m_progressMessage, progressChar.repeat(numDots), progressRemainingChar.repeat(progressBarSize - numDots)));

        if (numDots == progressBarSize) {
            System.err.println("\nDone!");
        }
    }

    @Override
    public void show() {

    }

    @Override
    public void hide() {

    }
}
