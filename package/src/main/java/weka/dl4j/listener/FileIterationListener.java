/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    FileIterationListener.java
 *    Copyright (C) 2016 University of Waikato, Hamilton, New Zealand
 *
 */
package weka.dl4j.listener;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;

import lombok.Builder;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;

/**
 * Class for listening to performance stats and writing them to a file.
 *
 * @author Christopher Beckham
 */
public class FileIterationListener extends weka.dl4j.listener.EpochListener {

    /**
     * The ID used for serializing this class
     */
    private static final long serialVersionUID = 1948578564961956518L;

    /**
     * The print writer to use
     */
    private transient PrintWriter m_pw = null;

    /**
     * Constructor for this listener.
     *
     * @param filename the file to write the information to
     * @throws Exception
     */
    public FileIterationListener(String filename) throws Exception {
        super();
        File f = new File(filename);
        if (f.exists()) f.delete();
        System.out.println("Creating debug file at: " + filename);
        m_pw = new PrintWriter(new FileWriter(filename, false));
    }

    @Override
    public void log(String msg) {
        m_pw.write(msg + "\n");
        m_pw.flush();
    }
}