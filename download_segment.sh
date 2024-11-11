echo "Downloading segment $1"

# Download the .zarr segment
rclone copy scrolls_remote:$1.zarr/ ./$1.zarr/ --progress --multi-thread-streams=32 --transfers=32 --size-only
# Download the mask
wget https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths/$1/$1_mask.png
mv $1_mask.png $1.zarr/$1_mask.png

# Download the inklabels
rclone copy scrolls_remote:$1_inklabels.png ./$1.zarr/ --progress --multi-thread-streams=32 --transfers=32 --size-only

mkdir data
mv $1.zarr data/$1.zarr