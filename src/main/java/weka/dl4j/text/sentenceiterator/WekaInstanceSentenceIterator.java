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
 *    WekaInstanceSentenceIterator.java
 *    Copyright (C) 1999-2017 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.dl4j.text.sentenceiterator;

import org.deeplearning4j.text.sentenceiterator.BaseSentenceIterator;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Iterator;

/**
 * A Deeplearning4j's sentence iterator for Weka Instances. It considers Strings from a given
 * attribute index. If the String attribute has multiple lines, they are splitted.
 *
 * @author Felipe Bravo-Marquez
 */
public class WekaInstanceSentenceIterator extends BaseSentenceIterator {

  /** The Weka Instances object to process */
  private Instances instances;

  /** The attribute String index */
  private int index;

  /** The Weka Instance iterator */
  private Iterator<Instance> iterator;

  /** An internal iterator for string attributes with multiple lines */
  private Iterator<String> internalIt;

  /** A flag indicator for string attributs with multiple lines */
  private boolean interItOn;

  /**
   * initializes the Object
   *
   * @param instances the Weka Instances object
   * @param index the attribute index
   */
  public WekaInstanceSentenceIterator(Instances instances, int index) {
    this.instances = instances;
    this.index = index;
    this.iterator = this.instances.iterator();
    this.interItOn = false;
  }

  /**
   * Gets the next String from a Weka Instance
   *
   * @return the String value
   */
  public String getNextWekaString() {
    Instance inst = this.iterator.next();
    return inst.stringValue(this.index);
  }

  /* (non-Javadoc)
   * @see org.deeplearning4j.text.sentenceiterator.SentenceIterator#hasNext()
   */
  @Override
  public boolean hasNext() {
    if (this.interItOn) return this.iterator.hasNext() || this.internalIt.hasNext();
    else return this.iterator.hasNext();
  }

  /* (non-Javadoc)
   * @see org.deeplearning4j.text.sentenceiterator.SentenceIterator#nextSentence()
   */
  @Override
  public String nextSentence() {
    if (!this.interItOn) {
      String line = getNextWekaString();
      String[] intLines = line.split("\n+");
      if (intLines.length > 1) {
        this.internalIt = Arrays.asList(intLines).iterator();
        this.interItOn = true;
        return this.internalIt.next();
      }
      return line;

    } else {
      if (this.internalIt.hasNext()) return this.internalIt.next();
      else {
        this.interItOn = false;
        String line = getNextWekaString();
        String[] intLines = line.split("\n+");
        if (intLines.length > 1) {
          this.internalIt = Arrays.asList(intLines).iterator();
          this.interItOn = true;
          return this.internalIt.next();
        }
        return line;
      }
    }
  }

  /* (non-Javadoc)
   * @see org.deeplearning4j.text.sentenceiterator.SentenceIterator#reset()
   */
  @Override
  public void reset() {
    this.iterator = this.instances.iterator();
    this.internalIt = null;
    this.interItOn = false;
  }
}
