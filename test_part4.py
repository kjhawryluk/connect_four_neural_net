from Board import Board
from util import battle
from Player import Player
from TFSessionManager import TFSessionManager
import tensorflow as tf
import random


def evaluate_players(p1: Player, p2: Player, games_per_battle=100, num_battles=100,
                     randomize: bool = False, p1_name: str = 'Player 1', p2_name: str = 'Player2'):

    board = Board()

    p1_wins = []
    p2_wins = []
    draws = []
    game_number = []
    game_counter = 0

    TFSessionManager.set_session(tf.Session())
    TFSessionManager.get_session().run(tf.global_variables_initializer())

    for i in range(num_battles):
        # Allows randomization for training full model.
        first_play = bool(random.getrandbits(1))
        if not randomize or first_play:
            p1win, p2win, draw = battle(p1, p2, games_per_battle)
        else:
            p2win, p1win, draw = battle(p2, p1, games_per_battle)
        p1_wins.append(p1win)
        p2_wins.append(p2win)
        draws.append(draw)
        game_counter = game_counter + 1
        game_number.append(game_counter)

    TFSessionManager.set_session(None)
    return game_number, p1_wins, p2_wins, draws


import matplotlib.pyplot as plt
from RandomPlayer import RandomPlayer
from SimpleNNQPlayer import NNQPlayer
from MinMaxAgent import MinMaxAgent

tf.reset_default_graph()

nnplayer = NNQPlayer("QLearner3", learning_rate=0.01, win_value=100.0, loss_value=-100.0)
mm_player = MinMaxAgent()
rndplayer = RandomPlayer()

game_number_ran_first, p1_wins_ran_first, p2_wins_ran_first, draws_ran_first = evaluate_players(rndplayer, nnplayer, num_battles=1000, p1_name='Random Player', p2_name='NN Player')

# reset nn
nnplayer_reset = NNQPlayer("QLearner2", learning_rate=0.01, win_value=100.0, loss_value=-100.0)
game_number_nn_first, p1_wins_nn_first, p2_wins_nn_first, draws_nn_first = evaluate_players(nnplayer_reset, rndplayer,
                                                                                            num_battles=1000,
                                                                                            p1_name='NN Player',
                                                                                            p2_name='Random Player')
# reset nn
nnplayer_rnd = NNQPlayer("QLearner1", learning_rate=0.01, win_value=100.0, loss_value=-100.0)
game_number_random, p1_wins_random, p2_wins_random, draws_random = evaluate_players(rndplayer, nnplayer_rnd,
                                                                                                num_battles=1000,
                                                                                                randomize=True,
                                                                                                p1_name='Random Player',
                                                                                                p2_name='NN Player')

nnplayer_rnd.nn.save_net()
nnplayer2 = NNQPlayer("QLearner1", learning_rate=0.01, win_value=100.0, loss_value=-100.0, restore_net=True, training=False)

game_number_nn_reset, p1_wins_nn_reset, p2_wins_nn_reset, draws_nn_reset = evaluate_players(nnplayer2, rndplayer,
                                                                                            num_battles=1000,
                                                                                            p1_name='NN Player',
                                                                                            p2_name='Random Player')
# game_number, p1_wins, p2_wins, draws = evaluate_players(nnplayer, mm_player) #, num_battles = 20)
fig, axs = plt.subplots(4)
axs[0].plot(game_number_ran_first, draws_ran_first, 'r-', game_number_ran_first, p2_wins_ran_first, 'g-', game_number_ran_first,  p1_wins_ran_first, 'b-')
axs[1].plot(game_number_nn_first, draws_nn_first, 'r-', game_number_nn_first, p1_wins_nn_first, 'g-', game_number_nn_first, p2_wins_nn_first, 'b-')
axs[2].plot(game_number_random, draws_random, 'r-', game_number_random, p2_wins_random, 'g-', game_number_random, p1_wins_random, 'b-')
axs[3].plot(game_number_nn_reset, draws_nn_reset, 'r-', game_number_nn_reset, p1_wins_nn_reset, 'g-', game_number_nn_reset, p2_wins_nn_reset, 'b-')
try:
    print("saving fig")
    plt.savefig("comparison.pdf")
except:
    print("Could not save fig")
    plt.show()
