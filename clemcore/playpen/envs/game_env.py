import os
from copy import deepcopy
from typing import List, Tuple, Dict, Callable, Union

from clemcore.backends import Model
from clemcore.clemgame import GameBenchmark, DialogueGameMaster, GameInstanceIterator, DefaultGameRecorder, Player
from clemcore.clemgame.resources import store_file
from clemcore.playpen.envs import PlayPenEnv


class GameEnv(PlayPenEnv):

    def __init__(self, game: GameBenchmark, player_models: List[Model], task_iterator: GameInstanceIterator,
                 reset=True):
        super().__init__()
        self._game = game
        self._game_name = game.game_name
        self._player_models = player_models
        self._dialogue_pair_descriptor = game.get_dialogue_pair_descriptor(player_models)
        self._task_iterator = task_iterator
        if len(self._task_iterator) < 1:
            try:
                self._task_iterator.reset()
            except:
                raise RuntimeError(f"No game instances given for the game: '{self._game_name}'")
        # variables initialized on reset()
        self._game_instance: Dict = None
        self._experiment: Dict = None
        self._master: DialogueGameMaster = None
        if reset:  # if reset, then the game env is fully functional after init
            self.reset()

    def __deepcopy__(self, memo):
        _copy = type(self).__new__(self.__class__)
        memo[id(self)] = _copy
        _copy.__dict__.update(self.__dict__.copy())  # shallow copy of most attributes (for now)
        _copy._master = deepcopy(self._master)
        _copy._task_iterator = deepcopy(self._task_iterator)
        return _copy

    @property
    def initial_prompts(self):
        return [{player: player.initial_prompt} for player in self.master.get_players()]

    @property
    def experiment(self):
        return self._experiment

    @property
    def master(self):
        return self._master

    @master.setter
    def master(self, master):
        self._master = master

    def reset(self) -> None:
        try:
            self._experiment, self._game_instance = next(self._task_iterator)
            self.master = self._game.create_game_master(self._experiment, self._player_models)
            self.master.game_recorder = DefaultGameRecorder(self._game_name,
                                                            self._experiment["name"],
                                                            self._game_instance["game_id"],
                                                            self._dialogue_pair_descriptor)
            self.master.setup(**self._game_instance)
        except StopIteration:
            self._task_iterator.reset()
            self.reset()

    def observe(self, batch=False) -> Tuple[Union[Player, Callable], Union[Dict, List[Dict]]]:
        player = self.master.get_current_player()
        context = self.master.get_context_for(player)
        
        # full context is handled within the player. 
        # In order to get the full context we need to obtain the context/_messages from player
        # and add the context from the game master.
        # It'd be more suitable to call context here 'observation' as the true context is _messages + observation 
        if batch:
            context = player.prepare_model_input(context)
        return player, context

    def step(self, response: Union[str, List]) -> Tuple[Union[bool, List], Union[Dict, List]]:
        self._done, info = self.master.step(response)
        return self._done, info

    def store_records(self, top_dir: str, rollout_dir: str, episode_dir: str,
                      store_experiment: bool = False, store_instance: bool = False):
        experiment_dir = f"{self.experiment['index']}_{self.experiment['name']}"
        experiment_path = os.path.join(top_dir,
                                       self._dialogue_pair_descriptor,
                                       rollout_dir,
                                       self._game_name,
                                       experiment_dir)
        episode_path = os.path.join(experiment_path, episode_dir)
        if store_experiment:
            store_file(self.experiment, f"experiment_{self.experiment['name']}.json", experiment_path)
        if store_instance:
            store_file(self._game_instance, f"instance.json", episode_path)
        store_file(self.master.game_recorder.interactions, f"interactions.json", episode_path)
        store_file(self.master.game_recorder.requests, f"requests.json", episode_path)



class BatchEnv(PlayPenEnv):

    def __init__(self, game: GameBenchmark, player_models: List[Model], task_iterator: GameInstanceIterator,
                 reset=True, batch_size=4):
        super().__init__()
        self._game = game
        self._game_name = game.game_name
        self._player_models = player_models
        self._dialogue_pair_descriptor = game.get_dialogue_pair_descriptor(player_models)
        self._task_iterator = task_iterator
        self._batch_size = batch_size

        if len(self._task_iterator) < 1:
            raise RuntimeError(f"No game instances given for the game: '{self._game_name}'")

        self.envs = self.generate_envs()
        self.active_envs = list(range(self._batch_size))  # Track active environments

    def generate_envs(self):
        """
        Generate `batch_size` GameEnvs that will roll out in parallel.
        """
        envs = {}
        for i in range(self._batch_size):
            envs[i] = GameEnv(self._game, self._player_models, self._task_iterator)
        return envs

    def observe(self):
        """
        Collect observations from all active environments.
        Returns a dictionary with environment IDs as keys.
        """
        obs_dict = {}
        for key in self.active_envs:
            env = self.envs[key]
            player, context = env.observe(batch=True)
            obs_dict[key] = {"player": player, "context": context}
        return obs_dict

    def step(self, responses: Dict[int, Union[str, List]]):
        """
        Apply a batch of responses to the corresponding environments.
        Args:
            responses: A dictionary where keys are environment IDs and values are responses.
        Returns:
            A dictionary with environment IDs as keys and step results (`done`, `info`) as values.
        """
        info_dict = {}
        for key, response in responses.items():
            env = self.envs[key]
            done, info = env.step(response)
            info_dict[key] = {"done": done, "info": info}

            # # Reset the environment if it is done
            # if done:
            #     self.env_reset(key)

        return info_dict

    def env_reset(self, env_id: int):
        """
        Reset a specific environment by its ID and move it to the end of the stack.
        """
        if env_id in self.envs:
            self.envs[env_id].reset()
            # Remove the environment from the active stack if it exists
            if env_id in self.active_envs:
                self.active_envs.remove(env_id)
            # Add the environment back to the end of the stack
            self.active_envs.append(env_id)

    def reset_batch(self):
        """
        Reset all environments in the batch.
        """
        self.envs = self.generate_envs()
        self.active_envs = list(range(self._batch_size))  # Track active environments

    def get_env(self, env_id):
        return self.envs[env_id]

    def reset_iterator(self):
        # resets the iteration queue
        # doing this should always yield the same queue for evaluation
        self._task_iterator.reset()
        
    def reset(self):
        pass

    def store_records(self, top_dir: str, rollout_dir: str, episode_dir: str,
                  store_experiment: bool = False, store_instance: bool = False):
        
        experiment_dir = f"{self.experiment['index']}_{self.experiment['name']}"
        experiment_path = os.path.join(top_dir,
                                    self._dialogue_pair_descriptor,
                                    rollout_dir,
                                    self._game_name,
                                    experiment_dir)
        episode_path = os.path.join(experiment_path, episode_dir)
        if store_experiment:
            store_file(self.experiment, f"experiment_{self.experiment['name']}.json", experiment_path)
        if store_instance:
            store_file(self._game_instance, f"instance.json", episode_path)
        store_file(self.master.game_recorder.interactions, f"interactions.json", episode_path)
        store_file(self.master.game_recorder.requests, f"requests.json", episode_path)



class EvalBatchEnv(BatchEnv):

    def __init__(self, game: GameBenchmark, player_models: List[Model], task_iterator: GameInstanceIterator,
             reset=True, batch_size=4):
        """
        Initialize the EvalBatchEnv class.
        Args:
            game: The game benchmark instance.
            player_models: List of player models.
            task_iterator: The iterator providing game instances.
            reset: Whether to reset the environment on initialization.
            batch_size: Number of environments to run in parallel.
        """
        super().__init__(game, player_models, task_iterator, reset, batch_size)
        self.queue = self._task_iterator.get_queue()  # Get the full queue from the iterator
        self.len_active = min(len(self.queue), self._batch_size) # number of environments to activate
        print(f"Initialized len_active: {self.len_active}")  # Debug statement

        self.envs = self.generate_envs()  # Generate environments based on the queue
        self.active_envs = list(range(self.len_active))  # Track active environments

    def generate_envs(self):
        """
        Generate environments for all instances in the queue.
        """

        envs = {}
        for i in range(self.len_active):
            envs[i] = GameEnv(self._game, self._player_models, GameInstanceIterator([self.queue.pop(0)]))
        return envs
    
    def step(self, responses: Dict[int, Union[str, List]]):
        """
        Apply a batch of responses to the corresponding environments.
        Args:
            responses: A dictionary where keys are environment IDs and values are responses.
        Returns:
            A dictionary with environment IDs as keys and step results (`done`, `info`) as values.
        """
        info_dict = {}
        for key, response in responses.items():
            env = self.envs[key]
            done, info = env.step(response)
            info_dict[key] = {"done": done, "info": info}

            # If there are remaining items in the queue, create a new environment
            if done:
                self.shutdown_env(key)

                if len(self.queue) > 0:
                    new_env_id = max(self.envs.keys(), default=-1) + 1
                    self.envs[new_env_id] = GameEnv(
                        self._game,
                        self._player_models,
                        GameInstanceIterator([self.queue.pop(0)])
                    )
                    self.active_envs.append(new_env_id)
                    

        return info_dict

    def shutdown_env(self, env_id: int):
        """
        Remove a specific environment by its ID and shut it down.
        """
        if env_id in self.envs:
            del self.envs[env_id]  # Remove the environment from the dictionary
            if env_id in self.active_envs:
                self.active_envs.remove(env_id)  # Remove it from the active environments list
            print(f"Shutting down environment {env_id}")

    def reset_batch(self):
        """
        Reset all environments in the batch.
        """
        self.reset_iterator()
        self.queue = self._task_iterator.get_queue()  # Get the full queue from the iterator
        self.len_active = min(len(self.queue), self._batch_size) # number of environments to activate
        self.envs = self.generate_envs()  # Generate environments based on the queue
        self.active_envs = list(range( self.len_active))  # Track active environments


    def is_done(self):

        if self.len(self.active_envs) > 0:
            return False
        return True
    
    def get_rollout_length(self):

        return len(self.queue)