pip install --upgrade pip
pip install "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements
cp keys.py src/data_utils/keys.py
python src/data_utils/download_dataset_books2.py
tar -xvf  'data/processed/bookcorpus_shards.tar.gz' -C 'data/processed'