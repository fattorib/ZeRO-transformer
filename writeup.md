# How does Staged Sequence Training Interact with One Epoch Training?

Staged Sequence training involves initially training models with much shorter contexts at the beginning of training then increasing the context later in training (for example Shortformer begins training a model with contexts of 128 for 25% of the training steps and trains the rest at a context 3072 tokens). All the results I have seen for this type of training involve training models for multiple epochs. Bringing this approach to single epoch training raises two questions:

1. Do we still see the same benefits?
2. Can we reuse some of 'warmed' up sequences to further increase performance?

For question 2, during the warmup stage of training, we truncate the batches to reach the desired sequence length. In the case shown below, by utilizing a sequence warmup for 50% of the total training steps, we 'truncate out' almost 5.1B tokens! Could we retrain on these sequences at their full context and see furhter performance improvements?

## Background

We use the original OpenWebText corpus and add in all components of the OpenWebText2 corpus that were scraped after the creation of the original OWT corpus. Dataset is encoded with GPT2's tokenizer. Final dataset statistics:

```yaml
corpus: "openwebtext1+2"
tokenizer: "GPT2"
n_ctx: 1024
num_train_samples: 12574161
num_validation_samples: 806093
total_training_tokens: 12875940864 #~12B tokens
```


## Experiment Setup:

- All models are trained with a maximum context of 1024 tokens.
- Batch size of 512 is used during training
- Learning rate warmup over first 2000 batches from 0 to peak
- Learning rate decay follows cosine schedule to 10% of the peak over 90% of the remaining steps
- Final 10% of training steps are conducted at the final learning rate
- No dropout is used, only regularization comes from a weight decay of 0.1  

## Staged Sequence Training:

- When using contexts under 1024, all batches are truncated to the target context, keeping the number of sequences within a batch fixed. 
- Sequence length is increased over the first half of the dataset (~12k steps)
- For ease of implementation and to avoid retriggering XLA recompilation, sequence lengths were increased at 2 separate stages
- Shortest context of 128 was used for 6k steps at which it was doubled for 256 for another 6k steps. Once 12k steps were reached, the sequence length was increased to its maximum of 1024.

## Experiments Performed:

1. (**Baseline**): One epoch training at maximum context. *Tokens seen during training: 12.9B*
2. (**Experiment 1**): One epoch training with staged sequence length warmup as described above. *Tokens seen during training: 7.8B*
3. (**Experiment 2**): Staged sequence length warmup + total training step count adjusted (~+9.5K steps) to hit equal number of training tokens as baseline model. *Tokens seen during training: 12.9B*