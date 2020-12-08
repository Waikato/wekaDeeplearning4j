#!/usr/bin/env bash

if [[ $# -ne 2 ]]; then
    echo "Illegal number of parameters. Usage: ./version-bump.sh <old tag number> <new tag number>"
    echo "e.g.: ./version-bump.sh 1.7.0 1.7.1"
fi

old=$1
new=$2

sed -i.bak -e "0,/${old}/ s/${old}/${new}/" version
sed -i.bak -e "0,/${old}/ s/${old}/${new}/" cuda-scripts/install-cuda-libs.ps1
sed -i.bak -e "0,/${old}/ s/${old}/${new}/" cuda-scripts/install-cuda-libs.sh

git add version -v
git add cuda-scripts/install-cuda-libs.ps1 -v
git add cuda-scripts/install-cuda-libs.sh -v
git commit -m "Version bump v${old} to v${new}"
echo "Adding git tag: v$2"
git tag v$2
echo "Do not forget to push the new tag."
