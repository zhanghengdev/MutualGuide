#!/bin/sh

Copy the following script command.
start = 'date + %s'

echo "Prepare to download train-val2017 anotation zip file..."
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

echo "Prepare to download train2017 image zip file..."
wget -c http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

echo "Prepare to download test2017 image zip file..."
wget -c http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

end = 'date + %s'
runtime = $((end - start))

echo "Download completed in " $runtime  " second"

