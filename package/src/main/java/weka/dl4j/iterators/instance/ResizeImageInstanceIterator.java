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
 *    ImageDatasetIterator.java
 *    Copyright (C) 2016 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.dl4j.iterators.instance;

import org.datavec.api.split.CollectionInputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ResizeImageTransform;
import weka.core.Instances;
import weka.dl4j.ArffMetaDataLabelGenerator;

/**
 * An iterator that loads images and resizes them.
 *
 * @author Steven Lang
 */
public class ResizeImageInstanceIterator extends ImageInstanceIterator {

    /**
     * SerialVersionUID
     */
    private static final long serialVersionUID = -3310258401133869149L;

    /**
     * Default constructor with the new shape
     * @param iii Previous image iterator
     * @param newWidth New image width
     * @param newHeight New image height
     */
    public ResizeImageInstanceIterator(ImageInstanceIterator iii, int newWidth, int newHeight) {
        super();
        this.m_height = newHeight;
        this.m_width = newWidth;
        this.setTrainBatchSize(iii.getTrainBatchSize());
        this.setImagesLocation(iii.getImagesLocation());
        this.setNumChannels(iii.getNumChannels());
    }

    @Override
    protected ImageRecordReader getImageRecordReader(Instances data) throws Exception {
        ArffMetaDataLabelGenerator labelGenerator = new ArffMetaDataLabelGenerator(data, getImagesLocation().toString());
        ResizeImageTransform rit = new ResizeImageTransform(m_width, m_height);
        ImageRecordReader reader = new ImageRecordReader(m_height, m_width, getNumChannels(), labelGenerator, rit);
        CollectionInputSplit cis = new CollectionInputSplit(labelGenerator.getPathURIs());
        reader.initialize(cis);
        return reader;
    }
}
