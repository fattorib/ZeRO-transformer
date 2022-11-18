import webdataset as wds
import jax.numpy as jnp 
from transformers import GPTNeoXTokenizerFast
tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")


def preprocess(batch):
    x = batch["input_id.pth"]
    return jnp.array(x, dtype=jnp.int32)


train_dataset = wds.DataPipeline(
        wds.SimpleShardList('data/processed/sharded/openwebtext_validation-{000000..000001}.tar.gz'),
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.decode(handler=wds.warn_and_continue),
        wds.map(preprocess),
    )
count = 0 

for data in train_dataset:
    
    text = tokenizer.decode(data)
    
    with open(f'qa/text_{count}.txt', 'w') as f:
        f.write(text)
    count += 1