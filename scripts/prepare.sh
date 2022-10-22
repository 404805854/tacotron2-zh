mkdir data
rm ./data/* -rf
python merge_metadata.py
echo "merge metadata success"
python create_spk_info.py