package weka.dl4j;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;

public class FileIterationListener implements IterationListener {
	
	private static final long serialVersionUID = 1948578564961956518L;
	
	protected transient PrintWriter m_pw = null;
	
	public FileIterationListener(String filename) throws Exception {
		super();
		File f = new File(filename);
		if(f.exists()) f.delete();
		System.err.println("Creating debug file at: " + filename);
		m_pw = new PrintWriter(new FileWriter(filename, false));
		m_pw.write("epoch,loss\n");
	}

	@Override
	public void invoke() { }

	@Override
	public boolean invoked() {
		return false;
	}

	@Override
	public void iterationDone(Model model, int epoch) {
		System.err.println(epoch);
		m_pw.write( (epoch+1) + "," + model.score() + "\n");
		m_pw.flush();
	}	

}
