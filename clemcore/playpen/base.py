import abc

from clemcore.backends import Model
from clemcore.clemgame import GameRegistry
from clemcore.playpen.envs import PlayPenEnv
from clemcore.playpen.buffers import RolloutBuffer
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