# Hybrid Sharding

This branch explores the use of different JAX APIs such as the new ```xmap```


# Benchmarks
- Based off of https://github.com/kingoflolz/mesh-transformer-jax/blob/4c15ee74a8ce5d4bf2aee2462638c1b33c8288a8/tpuv38_example.py

BFLOAT16 benchmarks
```bash

    ZeRO Step - Global BS 512 - accum steps 8 - Num Executions 10
    Mesh Layout (dp): (8)
    Model Size: base
    Total Time: 49.3145s
    Param Count: 123787776
    Effective TFLOPS: 91.003
    MFU (%): 50.5572

    ZeRO Step - Global BS 512 - accum steps 32 - Num Executions 10
    Mesh Layout (dp): (8)
    Model Size: base
    Total Time: 41.8388s
    Param Count: 123787776
    Effective TFLOPS: 107.263
    MFU (%): 59.5907

```

Reference Values:
    TPU V2-8: 180 TFLOPS (Bfloat16)