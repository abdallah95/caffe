#!/bin/bash

# to be run from the current directory

bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir=$bash_dir
sub_dir=images/

for dataset in trainval test
do
  dst_file=$bash_dir/$dataset.txt
  if [ "$dst_file" ]
  then
    rm -f "$dst_file"
  fi

  echo "Create list for caltech $dataset..."
  dataset_dir=$root_dir/$dataset/$sub_dir/

  img_file=$bash_dir/$dataset"_img.txt"
  ls -1 "$dataset_dir" | sed -e 's/\$//' > "$img_file"
  sed -i "s/^/$dataset\/images\//g" "$img_file"
  sed -i "s/$//g" "$img_file"

  label_file=$bash_dir/$dataset"_label.txt"
  ls -1 "$dataset_dir" | sed -e 's/\.jpg$//' -e 's/\.png$//' > "$label_file"
  sed -i "s/^/$dataset\/annotations\//g" "$label_file"
  sed -i "s/$/.txt/g" "$label_file"

  paste -d' ' "$img_file" "$label_file" >> "$dst_file"

  rm -f "$label_file"
  rm -f "$img_file"


  # Generate image name and size infomation.
  if [ $dataset == "test" ]
  then
    "$bash_dir/../../build/tools/get_image_size" "$root_dir" "$dst_file" "$bash_dir/$dataset""_name_size.txt"
  fi

  # Shuffle trainval file.
  if [ $dataset == "trainval" ]
  then
    rand_file=$dst_file.random
    cat "$dst_file" | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > "$rand_file"
    mv "$rand_file" "$dst_file"
  fi
done
