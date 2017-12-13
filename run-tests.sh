#!/bin/bash

cd package
rm -r wekarefs

# Disable global maven options
MAVEN_OPTS="-ea -Xms1G -Xmx4G -Dorg.bytedeco.javacpp.maxbytes=12G -Dorg.bytedeco.javacpp.maxphysicalbytes=12G"
mvn test -P GPU -Dmaven.test.skip=false
cd ..
