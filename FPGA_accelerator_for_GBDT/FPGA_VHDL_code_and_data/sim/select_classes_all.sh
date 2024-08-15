#!/bin/bash

# $1 --> selected class num

# Check if the argument is valid
if ! [[ $1 =~ ^-?[0-9]+$ ]]; then
    echo "Invalid argument. Exiting."
    exit 1
fi

# Array of files
files=("IP_test.vhd" "KSC_test.vhd" "PU_test.vhd" "SV_test.vhd")

# Iterate over the files
for file in "${files[@]}"
do
    # Start and end of the pixels of every class
    start=$(grep -n "PIXELS OF CLASS" "$file" | grep " 0$" | cut -d ":" -f 1)
    end=$(($(grep -n "wait;" "$file" | cut -d ":" -f 1) - 1))

    # Remove the comments for every class
    sed -i ''${start}','${end}' s/^--//' "$file"

    if [ $1 -ge 0 ]; then
        # Comment every class
        sed -i ''${start}','${end}' s/^/--/' "$file"
        # Start and end of the pixels of the selected class
        start=$(grep -n "PIXELS OF CLASS" "$file" | grep " $1$" | cut -d ":" -f 1)
        new_end=$(grep -n "PIXELS OF CLASS" "$file" | grep " $(($1 + 1))$" | cut -d ":" -f 1)
        if [ "$new_end" ]
        then
            end=$new_end
        fi

        # Remove the comments for the selected class
        sed -i ''${start}','${end}' s/^--//' "$file"

        echo "Selected class $1 in $file"
    else
        echo "Selected every class in $file"
    fi

done

