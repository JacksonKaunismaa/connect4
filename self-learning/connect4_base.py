import os
import pickle
import re
from os.path import join
import numpy as np
import c4_nn_agent as c4_nn
import monte_carlo as mc

"""Sample implementation of a board game (in this case connect4) including procedures for moving, unmoving, checking for wins, etc.
The whole point of this project is that we can get the nn to learn how to play this game (in this example, connect4) and see if it can learn to
beat a simple 'stupid' connect 4 engine that uses the tradiotional alpha beta search approach to pick the 'best' move from a given state.
This AB engine uses no domain knowledge (cuz i suck at connect 4) apart from the rules of what a win/loss/draw is, but is nevertheless
very strong (you can try to play against it in the connect4_AB_human.exe), and was an interesting challenge for the NN to beat, using
tabula rasa reinforcement learning!"""


class BoardState(object):
    def __init__(self, white_black=None, board_heights=None):
        """board repr is an 2x8x8 array with 'front' board being current player, 'back/second' board being other player,
         playing is 0 if white's turn, 1 if black's turn, this is done so that """
        if white_black is not None:  # if you input a board state it will copy that position, not create a refernce
            self.heights = np.copy(board_heights)
            self.board = np.copy(white_black)
        else:
            self.heights = np.zeros(8).astype(
                np.int8)  # initialize the height of each column to 0 and the position to all zeros
            self.board = np.zeros((2, 8, 8)).astype(np.float16)

    def infer_heights(self):
        """If only the position of the pieces is known, infer_heights figures out what the heights should be so moves can be played
        properly (used for importing boards and playing around with them based only the string represetation of one)"""
        p1 = self.board[0, :, :] + self.board[1, :, :]  # combine white and black pieces
        p2 = [p1[:, col] for col in range(8)]  # split into columns
        h = []  # heights
        for col in p2:
            try:
                h.append(np.nonzero(col)[0][-1] + 1)
            except IndexError:
                h.append(0)
        self.heights = np.array(h)

    def __repr__(self):
        """This is what gets called when you want to print(BoardState), shows it as a proper connect 4 board, using W and B as the pieces
        however it must be noted that from move to move, the positions of all W and B pieces will switch, this is simply a peculiarity with
        how the game state is represented, so that if it is player 1's move, all the pieces he has played before will show up as W and
        all of player 2's pieces will be B, then when its player 2's turn, all his pieces will be W and player 1's, B"""
        border = "*" * 26  # top border
        the_str = ""  # build view row by row
        the_str += border + "\n"
        for x in range(7, -1, -1):  # go backwards cuz board representation is backward to how gravity should work
            the_str += "|"
            for y in range(8):  # dont mess with these random spaces, they are very finicky to make the board look good
                if self.board[0, x, y] == 1:
                    the_str += " W "
                elif self.board[1, x, y] == 1:
                    the_str += " B "
                else:
                    the_str += " - "
            the_str += "|\n"
        the_str += border
        return the_str

    def move_check(self, c):
        """Given a move in Column C, returns 1 if a win is detected (either side), 0 if nothing detected, and
        a value very close to 0 if a draw is detected (so it evaluates to True but is still essentially 0)"""
        try:
            h = self.heights[c] - 1
        except IndexError:
            return 0
        try:
            if h >= 3 and self.board[1, h - 3, c] and self.board[1, h - 2, c] and self.board[1, h - 1, c]:  # |
                return 1  # checks for column wins (getting 4 in a row in a column)
        except ValueError:
            print(self)
            print(c)
            print(self.heights.shape)
            print(self.heights)
            print(h)
            print(self.board[1, h - 3, c])
            print(self.board[1, h - 2, c])
            print(self.board[1, h - 1, c])
            print(self.board)
            raise

        dl_ur = 0  # /   # check the Down-Left to Up-Right diagonal (DLUR) for wins like /
        ul_dr = 0  # \   # check the Up-Left to Down-Right diagonal (ULDR) for wins like \
        lr = 0  # -      # check the row for wins like ----
        for i in range(-3, 4):  # this code is an 'optimized' mess, so don't even worry about trying to figure it out
            this_y = h + i  # just know that it checks if there is a row or either diagonal win
            this_x = c + i
            if this_x >= 0:
                try:
                    if self.board[1, h, this_x]:  # --
                        lr += 1
                        if lr >= 4:
                            return 1
                    else:
                        lr = 0
                except IndexError:
                    pass
            if this_y >= 0:
                that_x = c - i
                if this_x >= 0:
                    try:
                        if self.board[1, this_y, this_x]:
                            dl_ur += 1
                            if dl_ur >= 4:  # /
                                return 1
                        else:
                            dl_ur = 0
                    except IndexError:
                        pass
                if that_x >= 0:
                    try:
                        if self.board[1, this_y, that_x]:
                            ul_dr += 1
                            if ul_dr >= 4:  # \
                                return 1
                        else:
                            ul_dr = 0
                    except IndexError:
                        pass
        # if no win detected and there are no legal moves, return 1e-8 so it evaluates to True and ~=0 (very suspicious)
        if self.get_legal().size == 0:
            return 1e-8
        return 0

    def move(self, m):
        """Drop a piece into column m"""
        try:
            self.board[
                0, self.heights[m], m] = 1  # put a piece in the correct position, now it is the other player's turn
            self.board = self.board[::-1]  # swap dimensions so that the 'front' board is the current player
            self.heights[m] += 1
        except IndexError:
            print(self)
            print(f"Move error: cant put a piece in column {m}")
            raise

    def unmove(self, m):
        try:
            """Be very careful with this, as it can only reliably work when play a move and then immediately after
            call this method, otherwise the indexing will be off, but it is like magically lifting a piece out of the board from slot m"""
            # height = self.heights[m]
            self.heights[m] -= 1
            self.board[1, self.heights[
                m], m] = 0  # set 'back' board to 0 (remove the piece) so now its the other player's turn
            self.board = self.board[::-1]  # swap dimensions so that the 'front' board is the current player
        except IndexError:
            print(self)
            print(f"Unmove error: cant take away a piece in column {m}")
            raise

    def randomize(self, moves=15):
        """First empties board and heights, then plays 'moves' number of random moves, making sure none of them create 4 in a row by chance"""
        self.board = np.zeros((2, 8, 8))
        self.heights = np.zeros(8).astype(np.int8)
        assert (moves <= 63)  # cant play more than 64 moves on a 8x8 board
        for _ in range(moves):
            potential = np.random.choice(self.get_legal())  # pick a random move to play
            self.move(potential)
            while self.move_check(
                    potential):  # if that move led to a win by pure chance, unmove it and picka a different move
                self.unmove(potential)
                potential = np.random.choice(self.get_legal())
                self.move(potential)
        # noinspection PyUnboundLocalVariable
        return potential

    def to_board(self, str_board):
        """Given a board representatino string str(BoardState), turns it back into a BoardState object"""

        def repl_func(match_obj):  # regex to replace - with S and delete everything else
            if match_obj.group(0) == "-":
                return "S"
            else:
                return ""

        bad_chars = re.compile(r"\*|\||-|\n|\r| {1, 2}")
        str_board = re.sub(bad_chars, repl_func, str_board).replace(" ", "")  # remove all whitespace and - characters
        np_board = np.zeros(2 * 8 * 8).astype(np.float16)  # flattened board
        for it, char in enumerate(str_board):
            if char == "W":
                np_board[0 + it] = 1
            elif char == "B":
                np_board[64 + it] = 1
        np_board = np_board.reshape((2, 8, 8))
        np_board = np.array([np.flipud(np_board[0]), np.flipud(np_board[1])])
        self.board = np_board
        self.infer_heights()  # having built the np board back up, we can now infer_heights to get the correct heights

    def reverse(self):
        self_cpy = BoardState(white_black=self.board, board_heights=self.heights)
        self_cpy.board = self_cpy.board[::-1]
        return self_cpy

    def __hash__(self):  # unique identifier for a board state for referral in monte carlo tree search
        return hash(self.board.tostring())

    def get_legal(self):
        """Returns a ndarray like [0 1 2 5] if moves are available in columns 0, 1, 2, and 5"""
        return np.where(self.heights <= 7)[0]

    def human(self):
        ag = c4_nn.Connect4NN("./connect4_models")
        switch = False
        while self.get_legal().size != 0:
            print(self)
            if switch:
                p, v = ag.predict(self)
                print(list(p))
                print(v)
            switch = not switch
            m = input("move: ")
            if m[0] == 'u':
                self.unmove(int(m[1]))
            else:
                self.move(int(m))
                print(self.move_check(int(m)))


def parse_board(np_board):
    """Auxillary function to parse a BoardState (but actually takes an np board as input) into a format understandable by connect4_AB.exe
    so it can load the current board state and return an eval of that using its own algorithm (communication between the python BoardState
    and the one defined in connect4_AB.exe)"""
    board_str = ""
    for x in range(8):
        for y in range(8):
            if np_board[0, y, x]:
                board_str += "W"
            elif np_board[1, y, x]:
                board_str += "B"
            else:
                board_str += " "
    # board_str += "|"
    return board_str


def deduplicate(train_data):
    """takes duplicate boards in training data, and combines them into 1 sample by averaging both the reward and policy"""
    found = {}
    for sample in train_data:
        the_hash = hash(sample[0].tostring())  # keep a unique id for each board by its hash
        if the_hash not in found.keys():
            found[the_hash] = [0, sample]  # times_seen, sample
        else:
            found[the_hash][0] += 1  # num of times the position has been seen
            found[the_hash][1][1] = ((found[the_hash][1][1] * found[the_hash][0]) + sample[1]) / (
                found[the_hash][0] + 1)  # average policy
            found[the_hash][1][2] = ((found[the_hash][1][2] * found[the_hash][0]) + sample[2]) / (
                found[the_hash][0] + 1)  # average reward
    return [unique_sample[1] for unique_sample in
            found.values()]  # return a list of the deduplicated and averaged samples


def get_data():
    training_data = []  # get training data from past game generation iters
    games_loc = os.fsencode(
        join("datasetts", "c4_train_games"))  # load all training data files in the c4_train_games folder
    for file in os.listdir(games_loc):
        filename = os.fsdecode(file)
        with open(join("datasetts", "c4_train_games", filename), "rb") as f:
            training_data += list(pickle.load(f))
    return training_data


def analyze_data():
    training_data = get_data()
    training_data = deduplicate(training_data)  # deduplicate and shuffle data
    idxs = np.random.randint(len(training_data), size=50).astype(np.int64)
    training_data = np.array(training_data)

    for sample in training_data[:20]:
        print(BoardState(white_black=sample[0]))
        print(sample[1])
        print(sample[2])

    for sample in training_data[idxs, :]:
        print(BoardState(white_black=sample[0]))
        print(sample[1])
        print(sample[2])


def test():
    """Does a bunch of random testing of various BoardState functions as well as monte carlo tree search, the NN agent, etc."""
    g = BoardState()
    # g.human()
    # quit()
    g.to_board("""**************************
| -  -  -  -  -  -  -  - |
| -  -  -  -  -  -  -  - |
| -  -  -  -  -  -  -  - |
| -  -  -  -  -  -  -  - |
| -  -  -  W  -  -  -  - |
| -  -  B  B  -  W  -  - |
| W  W  W  B  -  B  W  - |
| B  B  W  B  B  W  B  - |
**************************""")  # turn board string into a BoardState
    other_g = BoardState()
    # other_g.move(1)
    # ag = c4_nn.Connect4NN("./datasetts", log_path="/tmp/tensorflow_logs/c4_why_eval9")  # load the nn agent
    # # res = ag.bucket_predict([other_g.board, g.board])
    # # print(res)
    # # print(res[1])
    # # print(res[1][0])
    # # quit()
    # # data = get_data()
    # # ag.train_update(data, epochs=5)
    # # ag.save()
    # ag.randomly_test()  # see c4_nn_agent.Connect4NN.randomly_test()
    # # tr = mc.MCTS()  # init game tree
    # # ag.train_update(get_data(), epochs=5)
    ag = c4_nn.Connect4NN("./connect4_models", log_path="/tmp/tensorflow_logs/c4_why_eval9")  # load the nn agent
    ag.play_human()
    # ag.randomly_test()
    # print(g)  # print the current board state before it gets evaluated using mcts
    # print(ag.predict(g))  # print the NN agent's initial prediciton of the policy
    # # do a bunch of iters, then show the tree statistics and the move selected, to test all relevant systems
    # tr.search_iter(g, ag, 0, iters=8000, threads=32)
    # move = np.random.choice(8, p=tr.get_policy(g, temperature=0.9))
    # while move not in g.get_legal():
    #     move = np.random.choice(8, p=tr.get_policy(g, temperature=0.9))
    # print("distr = ", list(tr.get_policy(g)))
    # print("chosen move: ", move)
    # print("eval: ", tr.get_eval(g))
    # print("q:", list(tr.Q[hash(g)]))
    # print("w:", list(tr.W[hash(g)]))
    # print("p:", list(tr.P[hash(g)]))
    # print("n:", list(tr.N[hash(g)]))


if __name__ == "__main__":
    test()
