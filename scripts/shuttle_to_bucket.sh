#!/bin/bash

BRANCH=bw-oopify
BUCKET_PATH=gs://fc-secure-94c57cb0-5039-4596-9a9e-03c1cde4cc15/cellmincer_tarballs
PWD=$(pwd)
TMP_PATH=${PWD}/__tmp__

mkdir -p ${TMP_PATH}
cd ${TMP_PATH}
git clone git@github.com:broadinstitute/CellMincer.git
cd CellMincer
git fetch --all
git checkout ${BRANCH}
COMMIT_ID=$(git rev-parse --short HEAD)
mkdir ../CellMincer_tmp
mv cellmincer REQUIREMENTS.txt setup.py MANIFEST.in README.md LICENSE ../CellMincer_tmp/
cd ..
rm -rf ./CellMincer
mv ./CellMincer_tmp ./CellMincer
tar --exclude-vcs -cvzf ./CellMincer.tar.gz ./CellMincer
gsutil cp ./CellMincer.tar.gz ${BUCKET_PATH}
cd ${PWD}
rm -rf ${TMP_PATH}


