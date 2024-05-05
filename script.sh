#!/bin/bash

root_dir="/home/mt0/22CS60R61/MTP/NipsPaper2023-8E26/MTP_2024/Braindecode/output"

# Use find to get a list of all txt files in the tree
find "$root_dir" -type f -name "*.txt" | while read -r txt_file; do
    # Your logic goes here for each txt file
    echo $txt_file
    sed -r "s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})?)?[mGK]//g" $txt_file >  $txt_file
done
