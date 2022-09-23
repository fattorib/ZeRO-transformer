# Running Code on a TPU VM (Pod)

While TPU VMs are great and easy to use in a single-host case, I found the documentation and provided JAX examples for TPU VM pods to be quite minimal (they pretty much show how to copy over a single file to all hosts and run it).

The commands detailed below were used on a TPU v3-32 VM with the following Jax versions:

```text
jax==0.3.17
jaxlib==0.3.15
libtpu-nightly==0.1.dev20220723
```

## Background

The basic command structure for running code on TPU VMs is as follows:

Copy over a file to all VM hosts (```--worker=all```):

```bash
gcloud compute tpus tpu-vm scp myscript my-tpu-name:  --worker=all --zone=my-tpu-zone
```

Run the file on all hosts:
```bash
gcloud compute tpus tpu-vm ssh my-tpu-name:  --worker=all --zone= my-tpu-zone --command="bash myscript.sh"
```

On (broken) option I found through googling is to paste in code blocks to the ```--command``` prompt. I was unable to get this working on my TPUs. 

## What I do 

I created separate bash scripts for each of the following tasks:

- Cloning repository and installing all dependances (```install.sh```)
- Testing code (```test.sh```)
- Updating code on hosts through ```git pull``` (```git.sh```)
- Running/Resuming training (```run.sh/resume.sh```)

I then create one main bash script that copies over all these scripts to the VM hosts and runs the installation. My script looks like:

setup_install.sh:
```bash
# copy over required scripts
gcloud compute tpus tpu-vm scp install.sh my-tpu-name:   --worker=all --zone=my-tpu-zone
gcloud compute tpus tpu-vm scp run.sh my-tpu-name:   --worker=all --zone=my-tpu-zone
gcloud compute tpus tpu-vm scp test.sh my-tpu-name:   --worker=all --zone=my-tpu-zone
gcloud compute tpus tpu-vm scp git.sh my-tpu-name:   --worker=all --zone=my-tpu-zone
gloud compute tpus tpu-vm scp resume.sh my-tpu-name: --worker=all --zone=my-tpu-zone--comma>

# run installation
gcloud compute tpus tpu-vm ssh my-tpu-name  --worker=all --zone=my-tpu-zone
```

Training on TPU is then as simple as:
```bash
gcloud compute tpus tpu-vm ssh my-tpu-name  --worker=all --zone=my-tpu-zone --command="bash run.sh"
```

## Quirks of TPU Pods:

Other issues I ran into and my solutions:

- I use [Weights and Biases](https://wandb.ai/) for all my model logging. Despite ensuring only one host interacts with the Weights and Biases run (by specifying ```jax.process_index() == 0```) we still require all hosts to be logged in or the other hosts will complain and the training will crash.

- I use [webdataset](https://github.com/webdataset/webdataset) and a PyTorch dataloader for all my data. To ensure proper splitting when training on multiple hosts, I created a modified ```webdataset.split_by_worker``` method to split the shards by Jax process:

```python
def split_by_jax_process(src):
    # get current hostid + total number of hosts
    host_id, num_process = (
        jax.process_index(),
        jax.device_count()//jax.local_device_count(),
    )
    if num_process > 1:
        for s in islice(src, host_id, None, num_process):
            yield s
    else:
        for s in src:
            yield s
```

Anytime you would use ```wds.split_by_worker``` replace it with the above method! The only thing you need to be aware of when using this fix is that each PyTorch dataloader can only have a single worker attached to it (done by setting ```num_workers = 0``` in your dataloaders).

- Interrupting training sessions with CTRL+C appears to cause an unrecoverable crash on TPU VMs and the VMs will no longer see the TPUs. If you need to cancel a run, one workaround I have found is to run a ```pkill``` on the ```python3``` process running on each host.

```bash 
gcloud compute tpus tpu-vm ssh my-tpu-name  --worker=all --zone=my-tpu-zone --command="pkill python3"
```

