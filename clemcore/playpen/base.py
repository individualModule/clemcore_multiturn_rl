import abc
import torch
from typing import Union

from clemcore.backends import Model
from clemcore.clemgame import GameRegistry
from clemcore.playpen.envs import PlayPenEnv
from clemcore.playpen.buffers import RolloutBuffer, BatchReplayBuffer, BatchRolloutBuffer
from clemcore.playpen.callbacks import CallbackList, BaseCallback


class BasePlayPen(abc.ABC):

    def __init__(self, learner: Model, teacher: Model):
        self.learner = learner
        self.teacher = teacher
        self.num_timesteps = 0
        self.callbacks = CallbackList()

    def add_callback(self, callback: BaseCallback):
        self.callbacks.append(callback)

    def _collect_rollouts(self, game_env: PlayPenEnv, rollout_steps: int, rollout_buffer: RolloutBuffer):
        # reset() sets up the next game instance;
        # we should notify somehow when all instances were run so users can intervene if wanted?
        self.callbacks.on_rollout_start(game_env, self.num_timesteps)
        rollout_buffer.initial_prompts = game_env.initial_prompts
        num_rollout_steps = 0
        while num_rollout_steps < rollout_steps:
            player, context = game_env.observe()
            response = player(context)
            done, info = game_env.step(response)
            num_rollout_steps += 1
            self.num_timesteps += 1
            rollout_buffer.on_step(context, response, done, info)
            self.callbacks.update_locals(locals())
            self.callbacks.on_step()
            if game_env.is_done():
                rollout_buffer.on_done()
                game_env.reset()
        self.callbacks.on_rollout_end()

    def is_learner(self, player):
        return player.model is self.learner

    def is_teacher(self, player):
        return player.model is self.teacher

    @abc.abstractmethod
    def learn_interactive(self, game_registry: GameRegistry):
        pass

class BatchRollout:

    def __init__(self, learner: Model, teacher: Model):
        """
        Initialize the BatchRollout class.

        Args:
            model: The model used for inference.
            callbacks: A list of callbacks to track rollout progress.
        """
        self.learner = learner 
        self.teacher = teacher
        self.num_timesteps = 0
        self.callbacks = CallbackList()

    def add_callback(self, callback: BaseCallback):
        self.callbacks.append(callback)

    @torch.no_grad()
    def _collect_rollouts(self, game_env: PlayPenEnv,
                        rollout_steps: int,
                        rollout_buffer: Union[BatchRolloutBuffer, BatchReplayBuffer],
                        forPlayer="Player 2",
                        ):
        """
        Collect rollouts using the BatchEnv, synchronizing turns for learner and teacher.

        Args:
            game_env: The BatchEnv instance managing multiple GameEnv instances.
            rollout_steps: The number of rollout steps to collect.
            rollout_buffer: The buffer to store collected trajectories.
            eval: Whether this is an evaluation rollout (no buffer flattening).
        """
        self.callbacks.on_rollout_start(game_env, self.num_timesteps)

        collected_trajectories = 0
        while collected_trajectories < rollout_steps:
            # game_env.align(self._remaining_trajectories(collected_trajectories, rollout_steps))  # Shut down surplus environments

            # Collect observations from all active environments
            observations = game_env.observe()

            # Separate observations by player type
            learner_inputs, learner_env_ids, teacher_inputs, teacher_env_ids = self._separate_inputs(observations)

            # Perform inference for learners
            learner_responses = self.learner.batch_generate(learner_inputs) if learner_inputs else []
            self._update_player_context(learner_env_ids, learner_responses, observations)
            # Perform inference for teachers
            teacher_responses = self.teacher.batch_generate(teacher_inputs) if teacher_inputs else []
            self._update_player_context(teacher_env_ids, teacher_responses, observations)
            print(f"Teacher resp: {len(teacher_responses)} --- Learner Resp: {len(learner_responses)}")
            # Combine responses for step processing
            responses = {env_id: response[2] for env_id, response in zip(learner_env_ids + teacher_env_ids, learner_responses + teacher_responses)}
            # Step through the environments with the responses
            step_results = game_env.step(responses)
            # Process step results
            for env_id, result in step_results.items():
                done = result["done"]
                info = result["info"]
                player = observations[env_id]["player"]

                # Add step to the rollout buffer only for the target player
                if self.learner_name in player.name:
                    # print('here')
                    self._add_step_buffer(env_id, player, info, done, rollout_buffer)
                    
                # If the environment is done, finalize the trajectory and reset
                if done:
                    if forPlayer in player.name:
                        collected_trajectories = self._update_rollout_state(game_env,
                                                                            env_id,
                                                                            rollout_buffer,
                                                                            collected_trajectories)
                        if collected_trajectories >= rollout_steps:
                            break

                    else:
                        # Drop the trajectory if it ended on the other player's turn
                        rollout_buffer.drop_trajectory(env_id)

                    # Reset the environment
                    game_env.env_reset(env_id)
            # Break the outer loop if the target number of trajectories is reached
            if collected_trajectories >= rollout_steps:
                break

        rollout_buffer.flatten_steps()
        self.callbacks.on_rollout_end()
        game_env.reset_batch()
        rollout_buffer.reset_active_trajectories()

    def _remaining_trajectories(self, rollout_steps, collected_trajectories):
        
        remaining = collected_trajectories - rollout_steps 
        return remaining

    def _separate_inputs(self, observations):

        # Separate observations by player type
        learner_inputs = []
        teacher_inputs = []
        learner_env_ids = []
        teacher_env_ids = []

        for env_id, obs in observations.items():
            player = obs["player"]
            context = obs["context"]
            if self.learner_name in player.name:
                learner_inputs.append(context)
                learner_env_ids.append(env_id)
            elif self.teacher_name in player.name:
                teacher_inputs.append(context)
                teacher_env_ids.append(env_id)

        return learner_inputs, learner_env_ids, teacher_inputs, teacher_env_ids
    
    def _update_player_context(self, env_ids, responses, observations):

        for env_id, response in zip(env_ids, responses):
            player = observations[env_id]["player"]
            context = observations[env_id]["context"]
            # player.update_context_and_response(context, response)
            player.process_model_output(context, response)

    def _add_step_buffer(self, env_id, player, info, done, rollout_buffer):

        full_context = player.get_context()[:-1]  # Retrieve full context excluding the last response
        # print(full_context)
        response_dict = player.get_context()[-1:]  # Get the last response as a list
        # print(response_dict)
        rollout_buffer.on_step(
            env_id=env_id,
            context=full_context.copy() if isinstance(full_context, dict) else full_context[:],
            response=response_dict.copy(),
            done=done,
            info=info.copy() if isinstance(info, dict) else info[:]
        )
    
    def _update_rollout_state(self, game_env, env_id, rollout_buffer, collected_trajectories):

        single_env = game_env.get_env(env_id)
        rollout_buffer.on_done(env_id)
        self.num_timesteps += 1
        collected_trajectories += 1
        self.callbacks.update_locals(locals())
        self.callbacks.on_step(single_env)

        return collected_trajectories



class EvalBatchRollout(BatchRollout):

    @torch.no_grad()
    def _collect_rollouts(self, game_env: PlayPenEnv,
                          rollout_buffer: Union[BatchRolloutBuffer, BatchReplayBuffer],
                          forPlayer="Player 2",
                          ):
        """
        Collect rollouts using the EvalBatchEnv, iterating over the entire queue.

        Args:
            game_env: The EvalBatchEnv instance managing multiple GameEnv instances.
            rollout_steps: The number of rollout steps to collect (ignored for EvalBatchEnv).
            rollout_buffer: The buffer to store collected trajectories.
            forPlayer: The player for which trajectories are collected.
            eval: Whether this is an evaluation rollout (default is True).
        """
        self.callbacks.on_rollout_start(game_env, self.num_timesteps)

        collected_trajectories = 0
        while not game_env.is_done():
            # Collect observations from all active environments
            observations = game_env.observe()

            # Separate observations by player type
            learner_inputs, learner_env_ids, teacher_inputs, teacher_env_ids = self._separate_inputs(observations)

            # Perform inference for learners
            learner_responses = self.learner.batch_generate(learner_inputs) if learner_inputs else []
            self._update_player_context(learner_env_ids, learner_responses, observations)

            # Perform inference for teachers
            teacher_responses = self.teacher.batch_generate(teacher_inputs) if teacher_inputs else []
            self._update_player_context(teacher_env_ids, teacher_responses, observations)

            # Combine responses for step processing
            responses = {env_id: response[2] for env_id, response in zip(learner_env_ids + teacher_env_ids, learner_responses + teacher_responses)}

            # Step through the environments with the responses
            step_results = game_env.step(responses)

            # Process step results
            for env_id, result in step_results.items():
                done = result["done"]
                info = result["info"]
                player = observations[env_id]["player"]

                # Add step to the rollout buffer only for the target player
                if forPlayer in player.name:
                    self._add_step_buffer(env_id, player, info, done, rollout_buffer)

                # If the environment is done, finalize the trajectory and shut it down
                if done:
                    if forPlayer in player.name:
                        collected_trajectories = self._update_rollout_state(game_env,
                                                                            env_id,
                                                                            rollout_buffer,
                                                                            collected_trajectories)
                    # Shut down the environment
                    game_env.shutdown_env(env_id)


        self.callbacks.on_rollout_end()
        game_env.reset_batch()
        rollout_buffer.reset_active_trajectories()
