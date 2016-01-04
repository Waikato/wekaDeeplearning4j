#!/bin/bash

ant -f build_package.xml make_package -Dpackage=ChrisDL4J
cd dist && java weka.core.WekaPackageManager -install-package ChrisDL4J0.0.1.zip
