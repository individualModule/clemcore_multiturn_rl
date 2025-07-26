from contextlib import contextmanager
from typing import List

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, benchmark
from clemcore.playpen.buffers import RolloutBuffer, BranchingRolloutBuffer, StepRolloutBuffer, ReplayBuffer, BatchReplayBuffer, BatchRolloutBuffer
from clemcore.playpen.callbacks import BaseCallback, GameRecordCallback, RolloutProgressCallback, CallbackList
from clemcore.playpen.base import BasePlayPen, BatchRollout, EvalBatchRollout
from clemcore.playpen.envs import PlayPenEnv
from clemcore.playpen.envs.game_env import GameEnv, BatchEnv, EvalBatchEnv
from clemcore.playpen.envs.branching_env import GameBranchingEnv

__all__ = [
    "BaseCallback",
    "GameRecordCallback",
    "RolloutProgressCallback",
    "CallbackList",
    "BasePlayPen",
    "BatchRollout",
    "EvalBatchRollout",
    "PlayPenEnv",
    "RolloutBuffer",
    "ReplayBuffer",
    "BranchingRolloutBuffer",
    "StepRolloutBuffer",
    "BatchReplayBuffer",
    "BatchRolloutBuffer",
    "GameEnv",
    "BatchEnv",
    "EvalBatchEnv",
    "GameBranchingEnv",
    "make_tree_env",
    "make_env",
    "make_batch_env"
]


@contextmanager
def make_env(game_spec: GameSpec, players: List[Model],
             instances_name: str = None, shuffle_instances: bool = False):
    with benchmark.load_from_spec(game_spec, do_setup=True, instances_filename=instances_name) as game:
        task_iterator = game.create_game_instance_iterator(shuffle_instances)
        yield GameEnv(game, players, task_iterator)


@contextmanager
def make_tree_env(game_spec: GameSpec, players: List[Model],
                  instances_name: str = None, shuffle_instances: bool = False,
                  branching_factor: int = 2, branching_model=None):
    with benchmark.load_from_spec(game_spec, do_setup=True, instances_filename=instances_name) as game:
        assert branching_factor > 1, "The branching factor must be greater than one"
        task_iterator = game.create_game_instance_iterator(shuffle_instances)
        yield GameBranchingEnv(game, players, task_iterator,
                               branching_factor=branching_factor, branching_model=branching_model)


@contextmanager
def make_batch_env(game_spec: GameSpec, players: List[Model],
                   instances_name: str = None, shuffle_instances: bool = False, batch_size: int = 4):
    
    with benchmark.load_from_spec(game_spec, do_setup=True, instances_filename=instances_name) as game:
        task_iterator = game.create_game_instance_iterator(shuffle_instances)
        yield BatchEnv(game, players, task_iterator, batch_size=batch_size)

@contextmanager
def make_eval_env(game_spec: GameSpec, players: List[Model],
                   instances_name: str = None, shuffle_instances: bool = False, batch_size: int = 4):
    
    with benchmark.load_from_spec(game_spec, do_setup=True, instances_filename=instances_name) as game:
        task_iterator = game.create_game_instance_iterator(shuffle_instances)
        yield EvalBatchEnv(game, players, task_iterator, batch_size=batch_size)