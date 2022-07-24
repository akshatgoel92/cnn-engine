#!/bin/bash

# Download and unzip dataset
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip

# Declare current variable as separate-folder in the current directory
current="$(pwd)/tiny-imagenet-200"

# Training data
cd $current/train
for DIR in $(ls); do
   cd $DIR
   rm *.txt
   mv images/* .
   rm -r images
   cd ..
done

# Validation data
cd $current/val
annotate_file="val_annotations.txt"
length=$(cat $annotate_file | wc -l)
for i in $(seq 1 $length); do
    # Fetch i-th line
    line=$(sed -n ${i}p $annotate_file)
    # Get file name and directory name
    file=$(echo $line | cut -f1 -d" " )
    directory=$(echo $line | cut -f2 -d" ")
    mkdir -p $directory
    mv images/$file $directory
done
rm -r images

# Remove zip file
cd ../..
rm -r *.zip

# Send done message
echo "Done..."
