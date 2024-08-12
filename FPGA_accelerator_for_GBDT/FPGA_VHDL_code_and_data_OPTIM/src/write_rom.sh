#!/bin/bash
# Check if the file argument is provided
if [ -z "$1" ]; then
    echo "Error: File argument is missing."
    exit 1
fi

file=$1

# Remove lines from rom.vhd
../sim/remove_lines.sh rom.vhd 32

# Append content to rom.vhd
cat ../class_roms/${file}_rom.txt >> rom.vhd

# Print success message
echo "ROM file successfully written to rom.vhd"