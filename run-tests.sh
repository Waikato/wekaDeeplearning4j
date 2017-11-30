#!/bin/bash

cd package
mvn test -P GPU -Dmaven.test.skip=false
cd ..
