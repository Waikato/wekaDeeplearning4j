package weka.core.progress;

import weka.gui.Logger;
import weka.gui.SysErrLog;
// TODO document
public class TextProgressBar extends AbstractProgressBar {

    public TextProgressBar(double maxProgress, String progressMessage) {
        super(maxProgress, progressMessage);
    }

    private String progressChar = "-";

    protected int progressBarSize = 60;

    private int numDots = 0;

    public void setProgress(double progress) {
        super.setProgress(progress);
        numDots = (int) (m_normalizedProgress * progressBarSize);

        refreshDisplay();
    }

    public void refreshDisplay() {
        System.err.print("\rProgress:  |" + progressChar.repeat(numDots) + " ".repeat(progressBarSize - numDots) + "|");

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
