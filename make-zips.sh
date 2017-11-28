#!/bin/bash

rm -r /tmp/zips
mkdir /tmp/zips 
./build.sh -c -v -b CPU
cp package/dist/*.zip /tmp/zips
./build.sh -c -v -b GPU
cp package/dist/*.zip /tmp/zips

