#!/bin/bash

rm -r /tmp/zips
mkdir /tmp/zips 
./build.sh -c -b CPU
cp package/dist/*.zip /tmp/zips
./build.sh -c -b GPU
cp package/dist/*.zip /tmp/zips

