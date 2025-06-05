import abc
import torch

from clemcore.backends import Model
from clemcore.clemgame import GameRegistry
from clemcore.playpen.envs import PlayPenEnv
from clemcore.playpen.buffers import RolloutBuffer, ReplayBuffer
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