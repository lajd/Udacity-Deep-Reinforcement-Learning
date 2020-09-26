from typing import List, Dict
from tools.rl_constants import Experience, BrainSet, Action
import torch
import numpy as np


def default_step_agents_fn(brain_set: BrainSet, next_brain_environment: dict, t: int):
    for brain_name, brain_environment in next_brain_environment.items():
        for i, agent in enumerate(brain_set[brain_name].agents):
            brain_agent_experience = Experience(
                state=brain_environment['states'][i].unsqueeze(0),
                action=brain_environment['actions'][i],
                reward=brain_environment['rewards'][i],
                next_state=brain_environment['next_states'][i].unsqueeze(0),
                done=brain_environment['dones'][i],
                t_step=t,
            )

            agent.step(brain_agent_experience)


def default_step_episode_agents_fn(brain_set: BrainSet, episode_number: int):
    for brain_name, _ in brain_set:
        for i, agent in enumerate(brain_set[brain_name].agents):
            agent.step_episode(episode_number)


def single_agent_step_agents_fn(brain_set: BrainSet, next_brain_environment: dict, t: int):
    for brain_name, brain_environment in next_brain_environment.items():
        agent = brain_set[brain_name].agents[0]
        brain_agent_experience = Experience(
            state=brain_environment['states'],
            action=brain_environment['actions'][0],
            reward=brain_environment['rewards'],
            next_state=brain_environment['next_states'],
            done=torch.LongTensor(brain_environment['dones']),
            t_step=t,
        )
        agent.step(brain_agent_experience)


def default_preprocess_brain_actions_for_env_fn(brain_actions: Dict[str, List[Action]]) -> Dict[str, List[Action]]:

    assert len(brain_actions) > 0 and isinstance(list(brain_actions.values())[0][0], Action), brain_actions

    outp = {}
    for brain, actions in brain_actions.items():
        if isinstance(actions, Action):
            outp[brain] = actions.value
        elif isinstance(actions, (tuple, list)):
            assert len(actions) > 0 and isinstance(actions[0].value, np.ndarray), actions[0].value
            outp[brain] = np.concatenate([i.value for i in actions], axis=0)
        else:
            raise ValueError("actions must be list of Action or Action, found: {}".format(type(actions)))
        assert isinstance(outp[brain], np.ndarray)

    return outp


class JointAttributes:
    def __init__(self, next_brain_environment: dict):
        all_states = []
        all_actions = []
        all_next_states = []
        for brain_name, brain_environment in next_brain_environment.items():
            all_states.extend(brain_environment['states'])
            all_actions.extend(brain_environment['actions'])
            all_next_states.extend(brain_environment['next_states'])

        self.all_states = all_states
        self.all_actions = all_actions
        self.all_next_states = all_next_states

    def get_joint_attributes(self):
        joint_states = torch.cat(self.all_states).view(1, -1)
        joint_actions = torch.from_numpy(np.concatenate([i.value for i in self.all_actions])).view(1, -1)
        joint_next_states = torch.cat(self.all_next_states).view(1, -1)
        return joint_states, joint_actions, joint_next_states


def multi_agent_step_agents_fn(brain_set: BrainSet, next_brain_environment: dict, t: int):
    joint_attributes = JointAttributes(next_brain_environment)
    joint_states, joint_actions, joint_next_states = joint_attributes.get_joint_attributes()
    for brain_name, brain_environment in next_brain_environment.items():
        for agent_number, agent in enumerate(brain_set[brain_name].agents):
            brain_agent_experience = Experience(
                state=brain_environment['states'][agent_number].unsqueeze(0),
                action=brain_environment['actions'][agent_number],
                reward=brain_environment['rewards'][agent_number],
                next_state=brain_environment['next_states'][agent_number].unsqueeze(0),
                done=brain_environment['dones'][agent_number],
                t_step=t,
                joint_state=joint_states,
                joint_action=joint_actions,
                joint_next_state=joint_next_states,
                brain_name=brain_name,
                agent_number=agent_number,
            )
            agent.step(brain_agent_experience)


def multi_agent_step_episode_agents_fn(brain_set: BrainSet, episode):
    for brain_name in brain_set.names():
        for _, agent in enumerate(brain_set[brain_name].agents):
            agent.step_episode(episode)
