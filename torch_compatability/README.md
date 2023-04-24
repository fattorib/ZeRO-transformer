# PyTorch Compatability Layer

This folder contains code to extract and convert a trained flax model to the corresponding
PyTorch model.

Assuming you have a trained flax state, run:

```bash
python torch_compatability/extract_msgpack.py --ckpt-dir *checkpoint directory* \
--prefix *exact name of flax trainstate* \

python torch_compatability/convert_to_torch.py --model-name *model name* \
--flax-path *path of extracted msgpack file* \
--torch-path *path to export torch model to * 
```

