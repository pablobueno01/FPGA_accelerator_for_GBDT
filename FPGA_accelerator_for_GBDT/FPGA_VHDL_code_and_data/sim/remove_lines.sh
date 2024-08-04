#!/bin/bash

# Check if the correct number of arguments is provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <file> <start_line> <end_line>"
    exit 1
fi

file="$1"
start_line="$2"
end_line="$3"

# Check if the file exists
if [ ! -f "$file" ]; then
    echo "File '$file' does not exist."
    exit 1
fi

# Check if the start line is greater than the end line
if [ "$start_line" -gt "$end_line" ]; then
    echo "Start line cannot be greater than end line."
    exit 1
fi

# Remove lines between start line and end line (inclusive)
sed -i "${start_line},${end_line}d" "$file"

echo "Lines between $start_line and $end_line (inclusive) have been removed from '$file'."