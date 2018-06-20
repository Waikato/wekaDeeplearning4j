#!/bin/bash

rm -r wekarefs

# Disable global maven options
MAVEN_OPTS="-ea -Xms1G -Xmx4G -Dorg.bytedeco.javacpp.maxbytes=12G -Dorg.bytedeco.javacpp.maxphysicalbytes=12G" mvn test -P cuda-9.1 -Dmaven.javadoc.skip=true -Dmaven.test.skip=false -Dweka.test.maventest=true
