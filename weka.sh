#!/bin/bash

java -Xmx5g -cp build/classes/:$WEKA_HOME/weka.jar:/Users/cjb60/github/weka-fresh/build/testcases/:lib/* weka.gui.GUIChooser
