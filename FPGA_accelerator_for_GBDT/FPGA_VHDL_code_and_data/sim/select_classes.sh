#!/bin/bash

# $1 --> selected class num

# Array of files
files=("IP_test.vhd" "KSC_test.vhd" "PU_test.vhd" "SV_test.vhd")

# Iterate over the files
for file in "${files[@]}"
do
    # Start and end of the pixels of every class
    start=$(grep -n "PIXELS OF CLASS" "$file" | grep " 0$" | cut -d ":" -f 1)
    end=$(($(grep -n "wait;" "$file" | cut -d ":" -f 1) - 1))

    # Remove the coments for every class
    sed -i ''${start}','${end}' s/^--//' "$file"

    # Comment every class
    sed -i ''${start}','${end}' s/^/--/' "$file"

    # Start and end of the pixels of the selected class
    start=$(grep -n "PIXELS OF CLASS" "$file" | grep " $1$" | cut -d ":" -f 1)
    new_end=$(grep -n "PIXELS OF CLASS" "$file" | grep " $(($1 + 1))$" | cut -d ":" -f 1)
    if [ "$new_end" ]
    then
        end=$new_end
    fi

    # Remove the coments for the selected class
    sed -i ''${start}','${end}' s/^--//' "$file"
done

