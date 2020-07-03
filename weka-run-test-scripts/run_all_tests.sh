#!/bin/bash

set -e

function print_msg {
  printf "\n\n##############\n"
  echo "$1"
  printf "##############\n"
}

for test_script in tests/*;
do
  print_msg "Running ${test_script}..."
  $test_script
done
