#
# Based on Tic-tac-toe game: https://github.com/fcarsten/tic-tac-toe
#

import numpy as np
from enum import Enum


class GameResult(Enum):
    """
    Enum to encode different states of the game. A game can be in progress (NOT_FINISHED), lost, won, or draw
    """
    NOT_FINISHED = 0
    BLACK_WIN = 1
    RED_WIN = 2
    DRAW = 3


#
# Values to encode the current content of a field on the board. A field can be empty, contain a red chip, or
# contain a black chip
#
EMPTY = 0  # type: int
BLACK = 1  # type: int
RED = 2  # type: int

#
# Define the length and width of the board. Has to be 3 at the moment, or some parts of the code will break. Also,
# the game mechanics kind of require this dimension unless other rules are changed as well. Encoding as a variable
# to make the code more readable
#
BOARD_HEIGHT = 6  # type: int
BOARD_WIDTH = 7  # type: int
BOARD_SIZE = BOARD_HEIGHT * BOARD_WIDTH  # type: int


class Board:
    """
    The class to encode a connect four board, including its current state of pieces.
    Also contains various utility methods.
    """

    #
    # We will use these starting positions and directions when checking if a move resulted in the game being
    # won by one of the sides.
    #
    WIN_CHECK_DIRS = [[(0, -1), (0, 1)],   # Horizontal
                      [(-1, -1), (1, 1)],  # Diagonal bottom left to top right
                      [(1, -1), (-1, 1)],   # Diagonal top left to bottom right
                      [(-1, 0)]            # Vertical
                      ]

    def hash_value(self) -> int:
        """
        Encode the current state of the game (board positions) as an integer. Will be used for caching evaluations
        :return: A collision free hash value representing the current board state
        """
        resStr = ''
        res = 0
        for i in range(BOARD_SIZE):
            #res *= 3
            resStr += str(self.state[i])

        return resStr

    @staticmethod
    def other_side(side: int) -> int:
        """
        Utility method to return the value of the other player than the one passed as input
        :param side: The side we want to know the opposite of
        :return: The opposite side to the one passed as input
        """
        if side == EMPTY:
            raise ValueError("EMPTY has no 'other side'")

        if side == RED:
            return BLACK

        if side == BLACK:
            return RED

        raise ValueError("{} is not a valid side".format(side))

    def __init__(self, s=None):
        """
        Create a new Board. If a state is passed in, we use that otherwise we initialize with an empty board
        :param s: Optional board state to initialise the board with
        """
        if s is None:
            self.state = np.ndarray(shape=(1, BOARD_SIZE), dtype=int)[0]
            self.reset()
        else:
            self.state = s.copy()

    def coord_to_pos(self, coord: (int, int)) -> int:
        """
        Converts a 2D board position to a 1D board position.
        Various parts of code prefer one over the other.
        :param coord: A board position in 2D coordinates
        :return: The same board position in 1D coordinates
        """
        return coord[0] * BOARD_HEIGHT + coord[1]

    def pos_to_coord(self, pos: int) -> (int, int):
        """
        Converts a 1D board position to a 2D board position.
        Various parts of code prefer one over the other.
        :param pos: A board position in 1D coordinates
        :return: The same board position in 2D coordinates
        """
        return pos // BOARD_HEIGHT, pos % BOARD_WIDTH

    def reset(self):
        """
        Resets the game board. All fields are set to be EMPTY.
        """
        self.state.fill(EMPTY)

    def num_empty(self) -> int:
        """
        Counts and returns the number of empty fields on the board.
        :return: The number of empty fields on the board
        """
        return np.count_nonzero(self.state == EMPTY)

    def random_empty_spot(self) -> int:
        """
        Returns a random empty spot on the board in 1D coordinates
        :return: A random empty spot on the board in 1D coordinates
        """
        while True:
            index = np.random.randint(self.num_empty())
            for i in range(BOARD_SIZE):
                if self.state[i] == EMPTY:
                    if index == 0:
                        if self.is_legal(i):
                            return i
                    else:
                        index = index - 1
        # top_filled_spots = self.get_top_disc_positions()
        # top_open_spots = list(spot + BOARD_WIDTH for spot in top_filled_spots if spot < BOARD_SIZE - BOARD_WIDTH)
        #
        # # Add empty columns
        # columns_used = list(pos % BOARD_WIDTH for pos in top_filled_spots)
        # for col in range(BOARD_WIDTH):
        #     if col not in columns_used:
        #         top_open_spots.append(col)
        #
        # # Choose a slot.
        # index = np.random.randint(0, len(top_open_spots))
        # return top_open_spots[index]

    def is_legal(self, pos: int) -> bool:
        """
        Tests whether a board position can be played. The spot has to be the first empty position in a column.
        :param pos: The board position in 1D that is to be checked
        :return: Whether the position can be played
        """
        return (0 <= pos < BOARD_SIZE) and (self.state[pos] == EMPTY) \
               and (pos < BOARD_WIDTH or self.state[pos - BOARD_WIDTH] != EMPTY)

    def move(self, position: int, side: int) -> (np.ndarray, GameResult, bool):
        """
        Places a piece of side "side" at position "position". The position is to be provided as 1D.
        Throws a ValueError if the position is not EMPTY
        returns the new state of the board, the game result after this move, and whether this move has finished the game

        :param position: The position where we want to put a piece
        :param side: What piece we want to play (RED, or BLACK)
        :return: The game state after the move, The game result after the move, Whether the move finished the game
        """
        if position > -1:
            if self.state[position] != EMPTY:
                print('Illegal move')
                raise ValueError("Invalid move")

            self.state[position] = side

        if self.check_win():
            return self.state, GameResult.RED_WIN if side == RED else GameResult.BLACK_WIN, True

        if self.num_empty() == 0:
            return self.state, GameResult.DRAW, True

        return self.state, GameResult.NOT_FINISHED, False


    def apply_dir(self, pos: int, direction: (int, int)) -> bool:
        """
        Applies 2D direction dir to 1D position pos.
        Returns the resulting 1D position, or -1 if the resulting position would not be a valid board position.
        Used internally to check whether either side has won the game.
        :param pos: What position in 1D to apply the direction to
        :param direction: The direction to apply in 2D
        :return: The resulting 1D position, or -1 if the resulting position would not be a valid board position.
        """
        row = pos // BOARD_WIDTH
        col = pos % BOARD_WIDTH
        row += direction[0]
        if row < 0 or row > BOARD_HEIGHT:
            return -1
        col += direction[1]
        if col < 0 or col > BOARD_WIDTH:
            return -1

        return row * BOARD_WIDTH + col


    def count_in_direction(self, pos: int, direction: (int, int)) -> int:
        '''
        Counts the number of chips that are the same color in a row.
        :param pos: Position of the chip just dropped.
        :param direction: Direction from the chip to check.
        :return Total count of chips in a given direction that are the same color as the starting chip:
        '''
        c = self.state[pos]
        count = 0
        if c == EMPTY:
            return count

        next_pos = self.apply_dir(pos, direction)

        while -1 < next_pos < BOARD_SIZE and self.state[next_pos] == c:
            count += 1
            next_pos = self.apply_dir(next_pos, direction)

        return count

    def check_win_in_dir(self, pos: int, direction: (int, int)) -> bool:
        """
        Checks and returns whether there are 3 pieces of the same side in a row if following direction dir
        Used internally to check whether either side has won the game.
        :param pos: The position in 1D from which to check if we have 3 in a row
        :param direction: The direction in 2D in which to check for 3 in a row
        :return: Whether there are 3 in a row of the same side staring from position pos and going in direction
        `direction`
        """
        c = self.state[pos]
        if c == EMPTY:
            return False

        p1 = int(self.apply_dir(pos, direction))
        p2 = int(self.apply_dir(p1, direction))

        if p1 == -1 or p2 == -1:
            return False

        if c == self.state[p1] and c == self.state[p2]:
            return True

        return False

    def get_top_disc_positions(self) -> [int]:
        """
        This goes through the positions in the top row and looks down the column until it finds a non-empty position.
        It adds that position to the array to return to get the highest disc positions in each column.
        :return: Array of positions of the highest discs for each column.
        """
        starting_top_pos = (BOARD_HEIGHT - 1) * BOARD_WIDTH
        top_disc_positions = []
        for top_pos in range(starting_top_pos, BOARD_SIZE):
            while top_pos >= BOARD_WIDTH and self.state[top_pos] == EMPTY:
                top_pos -= BOARD_WIDTH
            if top_pos >= 0:
                top_disc_positions.append(top_pos)

        return top_disc_positions

    def who_won(self) -> int:
        """
        Check whether either side has won the game and return the winner
        :return: If one player has won, that player; otherwise EMPTY
        """
        top_positions = self.get_top_disc_positions()
        for top_pos in top_positions:
            for win_direction in self.WIN_CHECK_DIRS:
                count_in_a_row = 0
                for direction in win_direction:
                    count_in_a_row += self.count_in_direction(top_pos, direction)
                    if count_in_a_row >= 4:
                        return self.state[top_pos]

        return EMPTY

    def check_win(self) -> bool:
        """
        Check whether either side has won the game
        :return: Whether a side has won the game
        """
        return self.who_won() != EMPTY

    def state_to_char(self, pos, html=False):
        """
        Return 'x', 'o', or ' ' depending on what piece is on 1D position pos. Ig `html` is True,
        return '&ensp' instead of ' ' to enforce a white space in the case of HTML output
        :param pos: The position in 1D for which we want a character representation
        :param html: Flag indicating whether we want an ASCII (False) or HTML (True) character
        :return: 'x', 'o', or ' ' depending on what piece is on 1D position pos. Ig `html` is True,
        return '&ensp' instead of ' '
        """
        if (self.state[pos]) == EMPTY:
            return '&ensp;' if html else ' '

        if (self.state[pos]) == BLACK:
            return 'b'

        return 'r'

    def html_str(self) -> str:
        """
        Format and return the game state as a HTML table
        :return: The game state as a HTML table string
        """
        data = self.state_to_charlist(True)
        html = '<table border="1"><tr>{}</tr></table>'.format(
            '</tr><tr>'.join(
                '<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in data)
        )
        return html

    def state_to_charlist(self, html=False):
        """
        Convert the game state to a list of list of strings (e.g. for creating a HTML table view of it).
        Useful for displaying the current state of the game.
        :param html: Flag indicating whether we want an ASCII (False) or HTML (True) character
        :return: A list of lists of character representing the game state.
        """
        res = []
        for i in range(3):
            line = [self.state_to_char(i * 3, html),
                    self.state_to_char(i * 3 + 1, html),
                    self.state_to_char(i * 3 + 2, html)]
            res.append(line)

        return res

    def __str__(self) -> str:
        """
        Return ASCII representation of the board
        :return: ASCII representation of the board
        """
        board_str = ""
        for i in range(3):
            board_str += self.state_to_char(i * 3) + '|' + self.state_to_char(i * 3 + 1) \
                         + '|' + self.state_to_char(i * 3 + 2) + "\n"

            if i != 2:
                board_str += "-----\n"

        board_str += "\n"
        return board_str

    def print_board(self):
        """
        Print an ASCII representation of the board
        """
        for i in range(3):
            board_str = self.state_to_char(i * 3) + '|' + self.state_to_char(i * 3 + 1) \
                        + '|' + self.state_to_char(i * 3 + 2)

            print(board_str)
            if i != 2:
                print("-----")

        print("")
