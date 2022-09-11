pip install --upgrade pip
pip install -r requirements
cp keys.py src/data_utils/keys.py
python src/data_utils/download_dataset.py
tar -xvf  'data/processed/bookcorpus_shards.tar.gz' -C 'data/processed'