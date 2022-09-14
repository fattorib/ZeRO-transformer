pip install --upgrade pip
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -r requirements
cp keys.py src/data_utils/keys.py
python src/data_utils/download_dataset.py
tar -xvf  'data/processed/bookcorpus_shards.tar.gz' -C 'data/processed'