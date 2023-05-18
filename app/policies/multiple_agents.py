from __future__ import absolute_import, division, print_function, annotations

import sys
from functools import partial
from typing import List, Optional, Callable

import reverb

import tensorflow as tf
from tf_agents.agents.tf_agent import TFAgent, LossInfo
from tf_agents.drivers import tf_driver
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step, time_step as ts
from tf_agents.typing import types
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils

from app.abstract.utils import MyCheckpointer
from app.models.pwr_market_env import PowerMarketEnv


class MultipleAgentsPolicy(TFPolicy):#TODO fix it
    def __init__(self, agents_list: List[Callable],
                 time_step_spec: ts.TimeStep,
                 action_spec: types.NestedTensorSpec,
                 policy_state_spec: types.NestedTensorSpec = (),
                 info_spec: types.NestedTensorSpec = ()):
        self._agents_list = agents_list#TODO do coefficient_functioniNonet properly. Add super and initialize proper TFPolicy


    def _action(self, time_step: ts.TimeStep, policy_state: types.NestedTensor,
                seed: Optional[types.Seed]) -> policy_step.PolicyStep:
        return policy_step.PolicyStep(
            action=tf.constant([agent.get_action(time_step.observation) for agent in self._agents_list], dtype=tf.float32),
            state=(),
            info=(),
        )

    def _distribution(self, time_step: ts.TimeStep, policy_state: types.NestedTensorSpec) -> policy_step.PolicyStep:
        pass

class MultipleTFAgents(TFAgent):
    def __init__(self,
                 agents_list: List,
                 time_step_spec,
                 action_spec,
                 **kwargs):
        self._agents_list = agents_list
        policy, collect_policy = self._setup_policy(time_step_spec, action_spec)
        super(MultipleTFAgents,self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy=policy,
            collect_policy=collect_policy,
            **kwargs)

    def _setup_policy(self,
                      time_step_spec: ts.TimeStep,
                      action_spec: types.NestedTensorSpec,
                      policy_state_spec: types.NestedTensorSpec = (),
                      info_spec: types.NestedTensorSpec = ()):

        policy = MultipleAgentsPolicy(
            [partial(agent.get_action, train_mode=True) for agent in self._agents_list],
            time_step_spec,
            action_spec,
            policy_state_spec,
            info_spec)

        collect_policy = MultipleAgentsPolicy(
            [partial(agent.get_action, train_mode=False) for agent in self._agents_list],
            time_step_spec,
            action_spec,
            policy_state_spec,
            info_spec)

        return policy, collect_policy


    def _train(self, experience: types.NestedTensor, weights: types.Tensor) -> LossInfo:
        loss = []
        extra = []
        for agent in self._agents_list:
            new_loss, new_extra = agent.train(agent.preprocess_sequence(experience))
            loss.append(new_loss)
            extra.append(new_extra)

        return LossInfo(loss, extra)

    def _loss(self, experience: types.NestedTensor, weights: types.Tensor) -> Optional[LossInfo]:
        pass


class MultipleAgents:
    num_iterations = 24 * 100 * 100
    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 24 * 100  # @param {type:"integer"} # 100 days in dataset

    initial_collect_steps = 24 * 5  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_capacity = 1000000  # @param {type:"integer"}

    batch_size = 24  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 24 * 10  # @param {type:"integer"}

    train_dir_base = "new_checkpoints"

    def __init__(self, train_env: PowerMarketEnv, eval_env: PowerMarketEnv, agents_list: List):
        self.py_train_env = train_env
        self.py_eval_env = eval_env

        self.train_env = TFPyEnvironment(train_env)
        self.eval_env = TFPyEnvironment(eval_env)

        self.global_step = tf.Variable(0)

        # self._action_spec = array_spec.BoundedArraySpec(
        #     shape=(), dtype=np.float32, minimum=-np.inf, maximum=np.inf, _name="action"
        # )
        # self._observation_spec = array_spec.BoundedArraySpec(
        #     shape=(13,), dtype=np.float32, minimum=-1., maximum=1., _name="observation"
        # )

        self._agent_list: List[SingleAgent] = agents_list
        self._number_of_agents = len(self._agent_list)

        self._agent = MultipleTFAgents(
            self._agent_list,
            time_step_spec=self.train_env.time_step_spec(),
            action_spec=self.train_env.action_spec(),
            train_sequence_length=2,
            train_step_counter=self.global_step,
            validate_args=False,
        )
        self._agent.initialize()

    @property
    def number_of_agents(self):
        return self._number_of_agents

    @property
    def agent(self):
        return self._agent


    def train(self):
        table_name = 'prioritised_table'
        replay_buffer_signature = tensor_spec.from_spec(
            self._agent.collect_data_spec)
        replay_buffer_signature = tensor_spec.add_outer_dim(
            replay_buffer_signature)
        table = reverb.Table(
            table_name,
            max_size=self.replay_buffer_capacity,
            sampler=reverb.selectors.Prioritized(0.8),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=replay_buffer_signature)
        checkpointer = reverb.checkpointers.DefaultCheckpointer(path=self.train_dir_base)

        reverb_server = reverb.Server([table], checkpointer=checkpointer)

        replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            self._agent.collect_data_spec,
            table_name=table_name,
            sequence_length=2,
            local_server=reverb_server)



        rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
            replay_buffer.tf_client,
            table_name,
            sequence_length=2,
        )

        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.batch_size,
            num_steps=2).prefetch(3)

        iterator = iter(dataset)


        self._agents_initialize_or_restore(self.train_dir_base)

        collect_driver = tf_driver.TFDriver(
            self.train_env,
            self._agent.collect_policy,
            [rb_observer],
            max_steps=self.collect_steps_per_iteration
        )

        time_step = self.train_env.reset()

        for _ in range(self.num_iterations):
            time_step, _ = collect_driver.run(time_step)

            experience, __ = next(iterator)
            train_loss = self._agent.train(experience).loss

            if _ % self.log_interval == 0:  # TODO make it show avg agent loss if possible
                remainder = _ % self.eval_interval
                percentage = ((remainder if remainder != 0 else self.eval_interval) * 70) // self.eval_interval
                sys.stdout.write("\r")
                sys.stdout.write("\033[K")
                sys.stdout.write(
                    f'[{"=" * percentage + " " * (70 - percentage)}] {self._name} self._loss: {train_loss} ')
                sys.stdout.flush()  # TODO denote agent's _name

            for agent in self._agent_list: agent.checkpointer.save(self.global_step.numpy())
            replay_buffer.py_client.checkpoint()














    def _agents_initialize_or_restore(self, dir: str = train_dir_base,):
        train_dir = '/'.join([
            dir,
            self.py_train_env.get_reward_function_name(),
            str(self._number_of_agents + '_Agents'),
            'Agent,'
        ])


        for agent in self._agent_list:
            agent.checkpointer = train_dir + str(agent._agent_id)
            if agent.checkpointer.checkpoint_exists:
                print("Agent-{0}: Checkpoint found. Initialized from checkpoint".format(agent._agent_id))
            agent.checkpointer.initialize_or_restore()




    # def __getattr__(self, item):
    #     return getattr(self._agent, item)









