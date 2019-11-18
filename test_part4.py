from Board import Board
from util import battle
from Player import Player
from TFSessionManager import TFSessionManager
import tensorflow as tf


def evaluate_players(p1: Player, p2: Player, games_per_battle=100, num_battles=100):
    board = Board()

    p1_wins = []
    p2_wins = []
    draws = []
    game_number = []
    game_counter = 0

    TFSessionManager.set_session(tf.Session())
    TFSessionManager.get_session().run(tf.global_variables_initializer())

    for i in range(num_battles):
        p1win, p2win, draw = battle(p1, p2, games_per_battle, False)
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

nnplayer = NNQPlayer("QLearner1", learning_rate=0.01, win_value=100.0, loss_value=-100.0)
mm_player = MinMaxAgent()
rndplayer = RandomPlayer()

game_number, p1_wins, p2_wins, draws = evaluate_players(mm_player, nnplayer, num_battles=1000)  # , num_battles = 20)
game_number_nn_first, p1_wins_nn_first, p2_wins_nn_first, draws_nn_first = evaluate_players(nnplayer, mm_player, num_battles=1000)  # , num_battles = 20)
# game_number, p1_wins, p2_wins, draws = evaluate_players(nnplayer, mm_player) #, num_battles = 20)
fig, axs = plt.subplots(2)
axs[0].plot(game_number, draws, 'r-', game_number, p1_wins, 'g-', game_number, p2_wins, 'b-')
axs[1].plot(game_number_nn_first, draws_nn_first, 'r-', game_number_nn_first, p1_wins_nn_first, 'g-', game_number_nn_first, p2_wins_nn_first, 'b-')

plt.show()
