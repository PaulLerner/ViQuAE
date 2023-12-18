#!/bin/bash

source_path="/home/lerner/code/meerqat/"
target_path="${LIA_CODE_REPO_ROOT}/meerqat/"


rsync -azh $source_path $target_path \
      --progress \
      --exclude=".git" --exclude-from=".gitignore"
