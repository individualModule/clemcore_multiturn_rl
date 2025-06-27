import abc
import torch
from typing import Union

from clemcore.backends import Model
from clemcore.clemgame import GameRegistry
from clemcore.playpen.envs import PlayPenEnv
from clemcore.playpen.buffers import RolloutBuffer, ReplayBuffer, BatchReplayBuffer, BatchRolloutBuffer
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
                        eval=False):
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
            # print(f'inputs')
            # print(learner_inputs)
            # print()

            # Perform inference for learners
            learner_responses = self.learner.batch_generate(learner_inputs) if learner_inputs else []
            self._update_player_context(learner_env_ids, learner_responses, observations)
            # print(learner_inputs)
            # print()
            # print(learner_responses)
            # print()
            # Perform inference for teachers
            teacher_responses = self.teacher.batch_generate(teacher_inputs) if teacher_inputs else []
            self._update_player_context(teacher_env_ids, teacher_responses, observations)
            print(f"Teacher resp: {len(teacher_responses)} --- Learner Resp: {len(learner_responses)}")
            # Combine responses for step processing
            responses = {env_id: response[2] for env_id, response in zip(learner_env_ids + teacher_env_ids, learner_responses + teacher_responses)}
            # Step through the environments with the responses
            step_results = game_env.step(responses)
            # print(observations)
            # print(responses)
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

        if not eval:
            # Flatten trajectories for further sampling
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


class BasePlayPenMultiturnTrajectory(BasePlayPen):
    """
    A Playpen class that collects rollouts and counts the number of collected trajectories
    instead of steps, while maintaining the multiturn logic from BasePlayPenMultiturn.
    """

    def _collect_rollouts(self, game_env: PlayPenEnv, rollout_steps: int, rollout_buffer: ReplayBuffer, forPlayer='Guesser', eval=False):
        # Notify callbacks that rollout is starting
        self.callbacks.on_rollout_start(game_env, self.num_timesteps)
        rollout_buffer.initial_prompts = game_env.initial_prompts

        collected_trajectories = 0
        retry_counter = 0
        retry_limit = 10
        while collected_trajectories < rollout_steps:
            with torch.no_grad():
                if retry_counter > retry_limit:
                    print('rollout terminated early!')
                    break
                player, context = game_env.observe()
                response = player(context)  # Returns a string - we don't want that
                done, info = game_env.step(response)

                # Retrieve the full context for the turn
                full_context = player.get_context()[:-1]  # Ensure this is unique for each step
                response_dict = player.get_context()[-1:]  # Get the player response only. Return as list

                # Add to buffer only if the player's name matches `forPlayer`
                if forPlayer in player.name:

                    rollout_buffer.on_step(
                        context=full_context.copy() if isinstance(full_context, dict) else full_context[:],
                        response=response_dict.copy(),
                        done=done,
                        info=info.copy() if isinstance(info, dict) else info[:]
                    )


                # Check if the game is done (trajectory completed)
                if game_env.is_done():
                    # Only collect the trajectory if the game ended on the desired player's turn
                    if forPlayer in player.name:
                        retry_counter = 0
                        rollout_buffer.on_done()
                        collected_trajectories += 1  # Increment trajectory count
                        self.num_timesteps += 1
                        self.callbacks.update_locals(locals())
                        self.callbacks.on_step()
                    else:
                        # Skip this trajectory if it ended on the other player's turn
                        rollout_buffer.drop_trajectory()  # Clear the buffer for this trajectory
                        print(f'Game end caused by other player. Dropping trajectory')
                        retry_counter +=1

                    game_env.reset()

        if not eval:
            # flatten the trajectories for further sampling.
            print('Rollout done - flattening trajectories')
            rollout_buffer.flatten_steps()
        # Notify callbacks that rollout has ended
        self.callbacks.on_rollout_end()


# consider collecting trajectories rather than steps - define the num of trajectories to sample.
class BasePlayPenMultiturn(BasePlayPen):
    """
    Base Playpen with a changed _collect_rollouts class to support multiturn context (as opposed to currently single turn).
    Also needs to differentiate between the number of players in the game.

    Pass the game-specific name of the player you want to collect rollouts for.
    """

    def _collect_rollouts(self, game_env: PlayPenEnv, rollout_steps: int, rollout_buffer: RolloutBuffer, forPlayer='Guesser'):
        # Notify callbacks that rollout is starting
        self.callbacks.on_rollout_start(game_env, self.num_timesteps)
        rollout_buffer.initial_prompts = game_env.initial_prompts

        num_rollout_steps = 0
        while num_rollout_steps < rollout_steps:
            with torch.no_grad():
                player, context = game_env.observe()
                response = player(context) # returns a string - we don't want that
                done, info = game_env.step(response)

                # Retrieve the full context for the turn
                full_context = player.get_context()[:-1]  # Ensure this is unique for each step
                response_dict = player.get_context()[-1:] # get the player response only. Return as list
                # Add to buffer only if the player's name matches `forPlayer`
                if forPlayer in player.name:

                    self.num_timesteps += 1
                    # only count rollout step if it's for the player we are trainign? 
                    num_rollout_steps += 1

                    rollout_buffer.on_step(
                        context=full_context.copy() if isinstance(full_context, dict) else full_context[:],
                        response=response_dict.copy(),
                        done=done,
                        info=info.copy() if isinstance(info, dict) else info[:]
                    )

                self.callbacks.update_locals(locals())
                self.callbacks.on_step()

                if game_env.is_done():
                    rollout_buffer.on_done()
                    game_env.reset()

        # Notify callbacks that rollout has ended
        self.callbacks.on_rollout_end()