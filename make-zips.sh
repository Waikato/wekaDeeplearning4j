#!/bin/bash

rm -r /tmp/zips
mkdir /tmp/zips 
./build.sh -c -v -b CPU
cp package/dist/*.zip /tmp/zips
./build.sh -c -v -b GPU
cp package/dist/*.zip /tmp/zips
old=$(pwd)
cd /tmp/zips

sumfile="/tmp/zips/sums.sha256"
echo "### SHA256 sums" >> ${sumfile}
echo "| sha256sum | file |" >> ${sumfile}
echo "| --- | --- |" >> ${sumfile}
for f in *.zip;
do	
	sum=$(sha256sum ${f} | cut -d" " -f1)
	echo "| ${f} | ${sum} |" >> /tmp/zips/sums.sha256
done
cd ${old}

