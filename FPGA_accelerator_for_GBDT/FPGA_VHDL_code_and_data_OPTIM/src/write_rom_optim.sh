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

# Remove lines from rom_centroids.vhd
../sim/remove_lines.sh rom_centroids.vhd 30

# Append content to rom_centroids.vhd
cat ../centroids_roms/${file}_rom_centroids.txt >> rom_centroids.vhd

# Print success message
echo "ROM file successfully written to rom_centroids.vhd"