#!/bin/bash

# Check if the correct number of arguments is provided
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <file> <start_line> [<end_line>]"
    exit 1
fi

file="$1"
start_line="$2"

# Check if the file exists
if [ ! -f "$file" ]; then
    echo "File '$file' does not exist."
    exit 1
fi

# Get the total number of lines in the file
total_lines=$(wc -l < "$file")

# Set end_line to the last line if not provided as an argument
if [ $# -eq 2 ]; then
    end_line="$total_lines"
else
    end_line="$3"
fi

# Check if the start line is greater than the end line
if [ "$start_line" -gt "$end_line" ]; then
    echo "Start line cannot be greater than end line."
    exit 1
fi

# Remove lines between start line and end line (inclusive)
if [ "$end_line" -eq "$total_lines" ]; then
    sed -i "${start_line},\$d" "$file"
else
    sed -i "${start_line},${end_line}d" "$file"
fi

echo "Lines between $start_line and $end_line (inclusive) have been removed from '$file'."