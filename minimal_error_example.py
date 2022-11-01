import functools
import os

import jax
import numpy as np
from typing import Any
from flax.training.train_state import TrainState
from jax.experimental.maps import Mesh
from jax.experimental.pjit import pjit
from jax.experimental import PartitionSpec
from src.models.GPT import model_getter
from src.utils.partitioning import create_opt_spec, set_partitions
from src.training.training_utils import get_optimizer
from main import train_step


from flax.training import train_state

class TrainState(train_state.TrainState):
    dynamic_scale: Any = None

def main():
    mesh_shape = (4, 2)
    devices = np.asarray(jax.devices()).reshape(*mesh_shape)
    mesh = Mesh(devices, ("dp", "mp"))


    model = model_getter("small", return_cfg=False, dtype = jax.numpy.bfloat16)


    rng = jax.random.PRNGKey(23)
    batch_tok = jax.random.randint(rng, shape=(1, 1024), maxval=50257, minval=0)
    param_shape = jax.eval_shape(model.init, rng, batch_tok)
    param_spec = set_partitions(param_shape)


    tx = get_optimizer(
        3e-4,0.01,model, 16, param_shape
    )

    def init_state(params):

        return TrainState.create(
            apply_fn=model.apply,
            tx=tx,
            params=params,
        )


    opt_state_shapes = jax.eval_shape(tx.init, param_shape)
    opt_state_spec = create_opt_spec(param_spec, opt_state_shapes)


    # create the state_spec given our opt and param specs
    state_spec = TrainState(
        params=param_spec,
        opt_state=opt_state_spec,
        tx=tx,
        step=None,
        apply_fn=model.apply,
    )

    train_step_pjit= pjit(train_step,
            in_axis_resources=(state_spec, PartitionSpec("dp"), None, None),
            out_axis_resources=(state_spec, None),
        )

    with mesh:

        init_batch = jax.numpy.ones(shape=(4, 1024), dtype=jax.numpy.int32)

        # shard params across mesh
        sharded_params = pjit(
            functools.partial(model.init, train=False),
            in_axis_resources=(None, None),
            out_axis_resources=(param_spec),
        )(rng, init_batch)

        state_sharded = pjit(
            init_state,
            in_axis_resources=(param_spec,),
            out_axis_resources=(state_spec),
        )(sharded_params)

        print('State Sharded Sucessfully')


        state,metrics = train_step_pjit(state_sharded, init_batch, None, None)

        print(metrics)

if __name__ == '__main__':
    main()