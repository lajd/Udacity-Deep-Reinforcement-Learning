from tasks.connect_n.connect_n_environment import ConnectN
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
# initialize our alphazero agent and optimizer
import torch.optim as optim
from tasks.connect_n import mcts
from copy import copy
import random


game_setting = {'size': (3, 3), 'N': 3}
game = ConnectN(**game_setting)


game.move((0,1))
print(game.state)
print(game.player)
print(game.score)

# player -1 move
game.move((0,0))
# player +1 move
game.move((1,1))
# player -1 move
game.move((1,0))
# player +1 move
game.move((2,1))

print(game.state)
print(game.player)
print(game.score)


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()

        '''
        # solution
        self.conv = nn.Conv2d(1, 16, kernel_size=2, stride=1, bias=False)
        self.size = 2*2*16
        self.fc = nn.Linear(self.size,32)

        # layers for the policy
        self.fc_action1 = nn.Linear(32, 16)
        self.fc_action2 = nn.Linear(16, 9)

        # layers for the critic
        self.fc_value1 = nn.Linear(32, 8)
        self.fc_value2 = nn.Linear(8, 1)
        self.tanh_value = nn.Tanh()
        '''
        self.conv = nn.Conv2d(1, 16, kernel_size=2, stride=1, bias=False)
        self.size = 2*2*16
        self.fc = nn.Linear(self.size,32)

        # layers for the policy
        self.fc_action1 = nn.Linear(32, 16)
        self.fc_action2 = nn.Linear(16, 9)

        # layers for the critic
        self.fc_value1 = nn.Linear(32, 8)
        self.fc_value2 = nn.Linear(8, 1)
        self.tanh_value = nn.Tanh()

    def forward(self, x):
        '''
        # solution
        y = F.relu(self.conv(x))
        y = y.view(-1, self.size)
        y = F.relu(self.fc(y))


        # the action head
        a = F.relu(self.fc_action1(y))
        a = self.fc_action2(a)
        # availability of moves
        avail = (torch.abs(x.squeeze())!=1).type(torch.FloatTensor)
        avail = avail.view(-1, 9)

        # locations where actions are not possible, we set the prob to zero
        maxa = torch.max(a)
        # subtract off max for numerical stability (avoids blowing up at infinity)
        exp = avail*torch.exp(a-maxa)
        prob = exp/torch.sum(exp)


        # the value head
        value = F.relu(self.fc_value1(y))
        value = self.tanh_value(self.fc_value2(value))
        return prob.view(3,3), value
        '''
        # solution
        y = F.relu(self.conv(x))
        y = y.view(-1, self.size)
        y = F.relu(self.fc(y))

        # the action head
        a = F.relu(self.fc_action1(y))
        a = self.fc_action2(a)
        # availability of moves
        avail = (torch.abs(x.squeeze())!=1).type(torch.FloatTensor)
        avail = avail.view(-1, 9)

        # locations where actions are not possible, we set the prob to zero
        maxa = torch.max(a)
        # subtract off max for numerical stability (avoids blowing up at infinity)
        exp = avail*torch.exp(a-maxa)
        prob = exp/torch.sum(exp)

        # the value head
        value = F.relu(self.fc_value1(y))
        value = self.tanh_value(self.fc_value2(value))
        return prob.view(3, 3), value


class Agent:
    def __init__(self):
        self.policy = Policy()

    def step(self, nn_v: float, nn_p: float, p: float, current_player: int) -> Tuple[float, float]:
        # solution
        # compute prob* log pi
        loglist = torch.log(nn_p)*p

        # constant term to make sure if policy result = MCTS result, loss = 0
        constant = torch.where(p > 0, p*torch.log(p), torch.tensor(0.))

        logterm = -torch.sum(loglist-constant)
        vterm = nn_v*current_player
        return vterm, logterm

    def step_episode(self, vterm, logterm, outcome) -> float:
        # '''
        # solution
        # # loss = torch.sum( (torch.stack(vterm)-outcome)**2 + torch.stack(logterm) )
        # optimizer.zero_grad()
        # '''
        loss = torch.sum((torch.stack(vterm)-outcome)**2 + torch.stack(logterm) )
        optimizer.zero_grad()
        loss.backward()
        # losses.append(float(loss))
        optimizer.step()
        return float(loss)

# policy = Policy()


def policy_player_mcts(game):
    mytree = mcts.Node(copy(game))
    for _ in range(50):
        mytree.explore(policy)

    mytreenext, (v, nn_v, p, nn_p) = mytree.next(temperature=0.1)

    return mytreenext.game.last_move


def random_player(game):
    return random.choice(game.available_moves())


game = ConnectN(**game_setting)
print(game.state)
policy_player_mcts(game)

# # % matplotlib notebook


# gameplay = Play(ConnectN(**game_setting),
#               player1=policy_player_mcts,
#               player2=policy_player_mcts)

game = ConnectN(**game_setting)
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=.01, weight_decay=1.e-4)

# train our agent


# episodes = 400
# outcomes = []
# losses = []
#
# widget = ['training loop: ', pb.Percentage(), ' ',
#           pb.Bar(), ' ', pb.ETA()]
# timer = pb.ProgressBar(widgets=widget, maxval=episodes).start()
#
# for e in range(episodes):
#
#     mytree = mcts.Node(ConnectN(**game_setting))
#     vterm = []
#     logterm = []
#
#     while mytree.outcome is None:
#         for _ in range(50):
#             mytree.explore(policy)
#
#         current_player = mytree.game.player
#         mytree, (v, nn_v, p, nn_p) = mytree.next()
#         mytree.detach_mother()
#
#         '''
#         # solution
#         # compute prob* log pi
#         loglist = torch.log(nn_p)*p
#
#         # constant term to make sure if policy result = MCTS result, loss = 0
#         constant = torch.where(p>0, p*torch.log(p),torch.tensor(0.))
#         logterm.append(-torch.sum(loglist-constant))
#
#         vterm.append(nn_v*current_player)
#         '''
#         # solution
#         # compute prob* log pi
#         loglist = torch.log(nn_p)*p
#
#         # constant term to make sure if policy result = MCTS result, loss = 0
#         constant = torch.where(p > 0, p*torch.log(p), torch.tensor(0.))
#         logterm.append(-torch.sum(loglist-constant))
#
#         vterm.append(nn_v*current_player)
#
#     # we compute the "policy_loss" for computing gradient
#     outcome = mytree.outcome
#     outcomes.append(outcome)
#
#     '''
#     solution
#     # loss = torch.sum( (torch.stack(vterm)-outcome)**2 + torch.stack(logterm) )
#     optimizer.zero_grad()
#     '''
#     loss = torch.sum((torch.stack(vterm)-outcome)**2 + torch.stack(logterm) )
#     optimizer.zero_grad()
#     loss.backward()
#     losses.append(float(loss))
#     optimizer.step()
#
#     if (e + 1) % 50 == 0:
#         print("game: ", e + 1, ", mean loss: {:3.2f}".format(np.mean(losses[-20:])),
#               ", recent outcomes: ", outcomes[-10:])
#     del loss
#
#     timer.update(e + 1)

# timer.finish()
#
#
# # plot your losses
#
#
# # % matplotlib notebook
# plt.plot(losses)
# plt.show()

# % matplotlib notebook

# # as first player
# gameplay=Play(ConnectN(**game_setting),
#               player1=None,
#               player2=policy_player_mcts)
#
#
# # % matplotlib notebook
#
# # as second player
#
# gameplay=Play(ConnectN(**game_setting),
#               player2=None,
#               player1=policy_player_mcts)
#
#
#
