#!/usr/bin/env bash

version=$(cat version)

rm windows-test/*zip
rm windows-test.zip
cp dist/wekaDeeplearning4j-${version}.zip windows-test/
cp dist/wekaDeeplearning4j-cuda-9.0-${version}-windows-x86_64.zip windows-test/
sed -i "s/{VERSION}/${version}/g" windows-test/start.bat

zip windows-test.zip windows-test/*

