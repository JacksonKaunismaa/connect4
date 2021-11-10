import os
import logging
logging.getLogger("tensorflow").disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import numpy as np
import tensorflow as tf
import connect4_base as cb
import monte_carlo as mc
import subprocess
import warnings
warnings.filterwarnings("ignore")

"""Actual NN agent that uses residual convolutional layers to compute a policy (probability distribution over the legal
moves of how likely that move is to lead to a win) and also a q score that can be interpreted as an expected score given
a Connect 4 board state (although this same architecture would work with other games, you'd just have to program the
rules of those games, and leave this part (relatively) untouched. The overall architecture is an initial convolutional
layer to scale the board to the right size and number of channels, then a bunch of residual layers (further description
below) and then another convolutional layer to scale board back down and to the right number of channels, and then 2
fully connected layers to flatten the input and combine the channels to produce 9 outputs (8 for the move
distributions, and 1 more for the expected score q)"""


class Connect4NN(object):
    def __init__(self, loc, fs=6, layers=11, log_path=None, device=0):
        device_name = f"/device:GPU:{device}"
        self.graph = tf.Graph()

        gpu_options = tf.GPUOptions()
        gpu_options.allow_growth = True
        gpu_options.visible_device_list = str(device)
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(graph=self.graph, config=config)

        self.fs = fs  # filter size for bottleneck layer
        self.layers = layers  # number of residual layers
        self.bottle = 64  # number of filters in bottleneck
        self.filters = 128  # number of filters between residual blocks
        self.fc_size = 512  # second last (fully connected) layer size
        self.dropout_p = 0.75  # dropout % for training

        # GRAPH/NN ARCHITECTURE
        with tf.device(device_name):
            with self.graph.as_default():
                with tf.name_scope("inputs"):
                    self.current_in = tf.placeholder(tf.float32, [None, 2, 8, 8], name="board_in")
                    self.board_shaped = tf.transpose(self.current_in, [0, 2, 3, 1],
                                                     name="board_shaped")
                    self.train_mode = tf.placeholder(tf.bool, name='train_mode')
                    self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
                    self.pi = tf.placeholder(tf.float64, [None, 8], name="improved_policy")
                    self.reward = tf.placeholder(tf.float32, [None], name="reward")
                    self.pkeep = tf.cond(self.train_mode, lambda: self.dropout_p, lambda: 1.0)

                with tf.name_scope("in_out_weights"):
                    # weights used in input layers and output layers (not in residual head)
                    self.scale_up = tf.Variable(tf.truncated_normal([5, 5, 2, self.filters], stddev=0.1), name=f'scaler_up')
                    self.scale_down = tf.Variable(tf.truncated_normal([2, 2, self.filters, self.bottle], stddev=0.1),
                                                  name=f'scaler_down')
                    # the 2 final fully connected layers' weights
                    self.fc1 = tf.Variable(tf.truncated_normal([4 * 4 * self.bottle, self.fc_size], stddev=0.1), name='fc1')
                    self.fc_v = tf.Variable(tf.truncated_normal([self.fc_size, 1], stddev=0.1), name='fc_v')
                    self.fc_p = tf.Variable(tf.truncated_normal([self.fc_size, 8], stddev=0.1), name='fc_p')

                def convolve_getter(name, shape):
                    # weight getter for residual layers
                    weight = tf.get_variable(name, shape=shape)  # by default uses glorot_uniform initializer (pretty good)
                    return weight

                def convolve_once(input_tensor, convolver):
                    # the most basic convolutional layer, including the convolution, batch norm, relu non-linearity
                    conv = tf.nn.conv2d(input_tensor, convolver, strides=[1, 1, 1, 1], padding="SAME")
                    conv_ = tf.layers.batch_normalization(conv, axis=-1, training=self.train_mode, scale=False, center=True)
                    return tf.nn.relu(conv_)

                def residual_block(input_tensor, layer_name):
                    # residual layer defintion, uses 3 parts, first reduces the number of channels, then some expensive
                    # convolutions using big filter sizes are done, then a final convolution to increase number of channels
                    with tf.variable_scope(layer_name):
                        save_input = input_tensor  # save input to be added to the output later (residual/skip connectino)
                        with tf.name_scope("weights"):
                            dec = convolve_getter("dec_channels", [1, 1, self.filters, self.bottle])
                            expensive = convolve_getter("convolve", [self.fs, self.fs, self.bottle, self.bottle])
                            inc = convolve_getter("inc_channels", [1, 1, self.bottle, self.filters])
                        with tf.name_scope("residual"):
                            p1 = convolve_once(input_tensor, dec)  # actually compute the 3-layered convolutions
                            p2 = convolve_once(p1, expensive)
                            p3 = convolve_once(p2, inc)
                        return tf.add(save_input, p3, name="act_out")  # residual connection

                with tf.name_scope("inc_channels"):
                    # first part of the architecture, scales up the input board_state to have more channels
                    scale_conv = tf.nn.conv2d(self.board_shaped, self.scale_up, strides=[1, 1, 1, 1], padding="SAME",
                                              name="scaling_up")
                    scale_bn = tf.layers.batch_normalization(scale_conv, axis=-1, training=self.train_mode,
                                                             scale=False, center=True, name="scaling_up_bn")
                    act = tf.nn.relu(scale_bn, name="scaling_up_act")
                    inc_act = act  # save for summaries

                for i in range(self.layers):  # all the residual layers
                    act = residual_block(act, f"layer-{i}")

                with tf.name_scope("dec_channels"):
                    # after residual layers, decrease the number of channels back down and reduce channel size
                    smaller = tf.nn.conv2d(act, self.scale_down, strides=[1, 2, 2, 1], padding="SAME", name="scaling_down")
                    smaller_bn = tf.layers.batch_normalization(smaller, axis=-1, training=self.train_mode,
                                                               scale=False, center=True, name="scaling_down_bn")
                    smaller_act = tf.nn.relu(smaller_bn, name="scaling_down_act")

                with tf.name_scope("fully_connected"):
                    # fully connected layers to reshape the convolutional channels into vectors
                    combine_filters = tf.matmul(tf.reshape(smaller_act, [-1, 4 * 4 * self.bottle]), self.fc1,
                                                name='combine_filters')  # scale down
                    combine_filters_bn = tf.layers.batch_normalization(combine_filters, axis=-1, training=self.train_mode,
                                                                       scale=False, center=True, name='combine_filters_bn')
                    combine_filters_a = tf.nn.relu(combine_filters_bn, name='combine_filters_a')
                    combine_filters_d = tf.nn.dropout(combine_filters_a, self.pkeep, name='combine_filters_d')

                with tf.name_scope("outputs"):
                    # final fully connected layer separated into a policy section and a value section
                    p_logits = tf.matmul(combine_filters_d, self.fc_p, name="policy_logits")
                    v_logits = tf.matmul(combine_filters_d, self.fc_v, name="value_logits")
                    self.weak_policy = tf.nn.softmax(tf.cast(p_logits, tf.float64), name="wp")  # move distribution
                    self.weak_value = tf.squeeze(tf.nn.tanh(v_logits, name='wv'))  # expected score in (-1, 1)

                with tf.name_scope("loss_opt"):  # for training, define loss and optimization operations
                    self.eval_loss = tf.losses.mean_squared_error(self.reward, self.weak_value)  # expected reward loss
                    self.p_loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.pi, logits=p_logits))
                    self.reg_loss = tf.add_n([tf.nn.l2_loss(w) for w in tf.trainable_variables()])  # regularization loss
                    self.loss = 2. * self.eval_loss + 1. * self.p_loss + 5e-5 * self.reg_loss  # weighted sum of 3 losses
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # batchnorm updates (alpha and beta)
                    with tf.control_dependencies(update_ops):  # as well as update population mean and std
                        # actual optimizer, using a learning rate of 0.0002 and somewhat high epsilon (usually its 1e-8)
                        self.optimizer = tf.train.AdamOptimizer(0.0002, name='opt', epsilon=1e-5).minimize(self.loss,
                                                                                                           global_step=self.global_step_tensor)

                # Useless summaries of weight/activation distributions and histograms for debugging purposes
                w_summaries = [tf.summary.histogram("inc_channels", self.scale_up),
                               tf.summary.histogram("dec_channels", self.scale_down),
                               tf.summary.histogram("fc_1", self.fc1),
                               tf.summary.histogram("fc_v", self.fc_v),
                               tf.summary.histogram("fc_p", self.fc_p)]

                # more summaries
                for i in range(self.layers):
                    with tf.variable_scope(f"layer-{i}", reuse=True):
                        with tf.name_scope("weights"):
                            dec_w = convolve_getter("dec_channels", [1, 1, self.filters, self.bottle])
                            exp_w = convolve_getter("convolve", [self.fs, self.fs, self.bottle, self.bottle])
                            inc_w = convolve_getter("inc_channels", [1, 1, self.bottle, self.filters])
                    layer_sum = [tf.summary.histogram(f"layer-{i}-dec", dec_w),
                                 tf.summary.histogram(f"layer-{i}-exp", exp_w),
                                 tf.summary.histogram(f"layer-{i}-inc", inc_w)]
                    w_summaries.append(layer_sum)
                self.summ_w_op = tf.summary.merge(w_summaries)

                act_summaries = [tf.summary.histogram("inc_channels_out", inc_act),
                                 tf.summary.histogram("residual_out", act),
                                 tf.summary.histogram("dec_channels_out", smaller_act),
                                 tf.summary.histogram("fc1_out", combine_filters_d),
                                 tf.summary.histogram("fc_v_out", v_logits),
                                 tf.summary.histogram("fc_p_out", p_logits),
                                 tf.summary.scalar("pkeep", self.pkeep)]
                self.summ_act_op = tf.summary.merge(act_summaries)

                train_summaries = [tf.summary.scalar("eval_loss", self.eval_loss),
                                   tf.summary.scalar("policy_loss", self.p_loss),
                                   tf.summary.scalar("reg_loss", self.reg_loss)]
                test_summaries = [tf.summary.scalar("eval_loss_t", self.eval_loss),
                                  tf.summary.scalar("policy_loss_t", self.p_loss),
                                  tf.summary.scalar("reg_loss_t", self.reg_loss)]
                self.summ_tr_op = tf.summary.merge(train_summaries)
                self.summ_te_op = tf.summary.merge(test_summaries)

                self.full_summ = tf.summary.merge(w_summaries + act_summaries)

                # saving/loading parts for model checkpointing and comparison
                self.saver = tf.train.Saver()
                self.loader = tf.train.Saver()
                if log_path is not None:
                    # pywriter = writer for any python variables you want to keep track of
                    # writer = writer for graph as well as activation and weight summaries
                    self.writer = tf.summary.FileWriter(log_path, self.sess.graph)
                    self.py_writer = tf.summary.FileWriter(log_path)
                self.sess.run(tf.global_variables_initializer())  # init graph
                try:
                    self.loader.restore(self.sess, tf.train.latest_checkpoint(os.path.join(os.path.dirname(__file__), loc)))  # load any models found
                except ValueError:
                    print("No models found, initializing random model...")

    def test_update(self, test_data, batch_size):
        if len(test_data) >= batch_size:
            batched_test = np.array_split(np.array(test_data), len(test_data) // batch_size)
        else:
            batched_test = [np.array(test_data)]
        for idx, te in enumerate(batched_test):
            tr_cost, summary = self.sess.run([self.loss, self.summ_te_op],
                                             feed_dict={self.current_in: np.stack(te[:, 0]),
                                                        self.train_mode: False,
                                                        self.pi: np.stack(te[:, 1]),
                                                        self.reward: np.stack(te[:, 2])})
            self.py_writer.add_summary(summary, self.get_step())

    def train_update(self, train_data, epochs=2, batch_size=32, test=0.005):
        """Train data should be a bunch of tuples of (board_state.board's, improved policy, diminish_game_reward), and this actually
        trains the nn on MCTS improved policy and expeced game reward"""
        random.shuffle(train_data)
        test_data = train_data[:int(len(train_data) * test)]  # set aside 0.5% of data as testing data
        train_data = train_data[int(len(train_data) * test):]
        print(len(test_data))
        print(len(train_data))
        for j in range(epochs):
            random.shuffle(train_data)  # randomly shuffle data and split data into batches
            if len(train_data) >= batch_size:
                batched_data = np.array_split(np.array(train_data), len(train_data) // batch_size)
            else:
                batched_data = [np.array(train_data)]
            cost = 0  # to record the cost for a batch
            pct = 0.0  # for the progress bar
            for idx, tr in enumerate(batched_data):
                o, batch_cost, summary = self.sess.run([self.optimizer, self.loss, self.summ_tr_op],
                                                       feed_dict={self.current_in: np.stack(tr[:, 0]),
                                                                  self.train_mode: True,
                                                                  self.pi: np.stack(tr[:, 1]),
                                                                  self.reward: np.stack(tr[:, 2])})
                cost += batch_cost
                # track the three loss types (see graph definition in __init__)
                self.py_writer.add_summary(summary, self.get_step())
                if float(idx) / len(batched_data) >= pct:  # progress bar
                    print("#", end='', flush=True)
                    pct += 0.025
                    self.test_update(test_data, batch_size)
            print(f"\nAverage cost on epoch {j}: {cost/len(train_data)}")  # test on the testing data to make sure cost is comparable

    def predict(self, inpt_game):
        """BoardState->policy, value, returns absolute prediction (without noise, isnan handling, and illegal moves)
        for a single board state"""
        policy, value = self.sess.run([self.weak_policy, self.weak_value],
                                      feed_dict={self.current_in: [inpt_game.board],
                                                 self.train_mode: False})
        return policy, value

    def save(self, loc=os.path.join(os.path.dirname(__file__), "connect4_models")):
        # save the current model to the specified location
        self.saver.save(self.sess, os.path.join(loc,
                                                f"agent-{self.get_step()}"))

    def kill(self):
        """This does not free GPU memory, or do anything useful at all, its actually 100% pointless, I thought at one
         point it might be useful but im pretty sure now that it actually does nothing (just in case, I've kept it)"""
        self.sess.close()

    def get_step(self):
        # get the iteration (model number)
        return tf.train.global_step(self.sess, self.global_step_tensor)

    def bucket_predict(self, game_boards):
        """given a bunch of boards in a 4d list of dimension num_boards x 2 x 8 x 8, it returns the policy and eval for each
         one, in order of its addition to the list, used in the APV-MCTS for batched gpu eval (see monte_carlo.py)"""
        # print("doing gpu eval size=", len(game_boards))
        # start = time.time()
        policies, values = self.sess.run([self.weak_policy, self.weak_value],
                                         feed_dict={self.current_in: np.array(game_boards),
                                                    self.train_mode: False})
        policies[np.isnan(
            policies)] = 0.125  # sometimes issues were happening with the weights getting too large and NaNs
        # print("gpu eval took", time.time() - start)  # its fixed now, so this is just here for safety
        return [policies, values]

    def tensorboard_log(self, kwargs=None):
        """Given a a dictionary of {name1:value1, name2:value2, ...}, it writes tensorboard log of those values with
        those names, and also writes a graph summary as well as some activation and weight summaries, its mostly just
        for debugging to make sure everything looks ok"""
        rand_board = cb.BoardState()
        rand_board.randomize()  # generate random board state for input so we can take a look at activations and stuff
        py_summary = tf.Summary()
        # summary = self.sess.run(self.summ_w_op)
        summary = self.sess.run(self.full_summ, feed_dict={self.current_in: [rand_board.board],
                                                           self.train_mode: False})
        # summary = self.sess.run(self.summ_act_op, feed_dict={self.current_in: [rand_board.board],
        #                                                    self.train_mode: False})
        if kwargs is not None:
            for n, v in kwargs.items():
                py_summary.value.add(tag=n, simple_value=v)  # write a summary for each item in the dictionary
        self.writer.add_summary(summary, self.get_step())  # log the graph and activation/weight summaries
        self.py_writer.add_summary(py_summary,
                                   self.get_step())  # log any additional summaries coming from the dictionary supplied

    def randomly_test(self, amount=20):
        """You can run this as a test to see if your nn is really learning anything or is just getting lucky, it
        generates a bunch of random board states and then feeds them through the nn and reports the eval and expected
        score, as you train your model more and more you should be able to see it assigning high probabilities to winning
        moves, but at first all values should be close to each other, approximately 12.5% for each move"""
        for _ in range(amount):
            rb = cb.BoardState()
            rb.randomize(np.random.randint(2, 17))  # create a board by randomly playing out anywhere from 2-17 moves
            print(rb)  # show the board
            praedicatus = self.predict(rb)  # calculate nn policy and expected score
            print("moves =", ["%.3f" % p for p in list(praedicatus[0][0])])  # display display policy and expected score (eval)
            print("eval =", praedicatus[1])

    def play_combo(self):
        import random
        switch = bool(random.randint(0,1))
        g = cb.BoardState()
        tr = mc.MCTS()
        pmove = 4
        temperature = 1.75
        for n in range(64):
            if switch:
                print(g.reverse())
                combo_move = combo_engine(self, tr, 2400, 80, 4.0, temperature, g, pmove)
                pmove = combo_move
                try:
                    print(f"eval = {tr.get_eval(g)}")
                except KeyError:
                    print("Unexpected move!")
                if g.move_check(combo_move) == 1:
                    print(g)
                    print("You lost!")
                    return
            else:
                print(g)
                try:
                    human_move = int(input("pick a move to play: "))
                except ValueError:
                    human_move = -1
                while human_move not in g.get_legal():
                    print("Invalid move")
                    try:
                        human_move = int(input("pick a move to play: "))
                    except ValueError:
                        pass
                g.move(human_move)
                try:
                    print(f"eval = {tr.get_eval(g)}")
                except KeyError:
                    print("Unexpected move!")
                pmove = human_move
                if g.move_check(human_move) == 1:
                    print("You won!")
                    print(g)
                    return
            switch = not switch
            if n == 7:
                temperature = 1e-5
        print("You got a draw!")

    def play_human(self):
        import random
        switch = bool(random.randint(0,1))
        g = cb.BoardState()
        tr = mc.MCTS()
        pmove = 0
        for _ in range(64):

            if switch:
                print(g.reverse())
                tr.search_iter(g, self, pmove, iters=1600, threads=80)
                chosen_move = np.random.choice(8, p=tr.get_policy(g))
                while chosen_move not in g.get_legal():
                    chosen_move = np.random.choice(8, p=tr.get_policy(g))
                print(f"eval = {tr.get_eval(g)}")
                g.move(chosen_move)
                pmove = chosen_move
                if g.move_check(chosen_move) == 1:
                    print(g)
                    print("You lost!")
                    return
            else:
                print(g)
                try:
                    human_move = int(input("pick a move to play: "))
                except ValueError:
                    human_move = -1
                while human_move not in g.get_legal():
                    print("Invalid move")
                    try:
                        human_move = int(input("pick a move to play: "))
                    except ValueError:
                        pass
                g.move(human_move)
                try:
                    print(f"eval = {tr.get_eval(g)}")
                except KeyError:
                    print("Unexpected move!")
                pmove = human_move
                if g.move_check(human_move) == 1:
                    print("You won!")
                    print(g)
                    return
            switch = not switch
        print("You got a draw!")


# noinspection PyPep8Naming
def eval_AB(board_pos, last_move):
    output = subprocess.run(  # get AB move
                    [os.path.join(os.path.dirname(__file__), "AB_with_eval.exe"), cb.parse_board(board_pos), "W", str(last_move)],
                    stdout=subprocess.PIPE)
    AB_eval = [int(v) for v in output.stdout.decode("utf-8").split()]  # get AB move, eval
    return AB_eval


def check_tactics(game_board, last_move):
    # noinspection PyPep8Naming
    AB_eval = eval_AB(game_board, last_move)   # evaluate once
    # print(AB_eval)
    return AB_eval


# noinspection PyPep8Naming
def AB_move(board_pos, last_move):
    output = subprocess.run(  # get AB move
        [os.path.join(os.path.dirname(__file__), "connect_4_AB.exe"), cb.parse_board(board_pos.board), "W", str(last_move)],
        stdout=subprocess.PIPE)
    numeric_move = int(output.stdout.decode("utf-8"))  # get AB move
    if numeric_move in board_pos.get_legal():  # make sure the AB move is legal, otherwise play a random legal move
        board_pos.move(numeric_move)
        return numeric_move
    else:
        rand_move = random.choice(board_pos.get_legal())
        board_pos.move(rand_move)
        return rand_move

def nn_search(agent, tree, search_iters, search_threads, c_val, temp, board_state, prev_move):
    """Given a board position, nn agent, search tree, and simulation parameters, it plays a move on the board and returns a tuple of
    (game_result, move_played) where game_result = 1 if the move was a win, else 0"""
    tree.search_iter(board_state, agent, prev_move, iters=search_iters, threads=search_threads, c_explore=c_val)
    chosen_move = np.random.choice(8, p=tree.get_policy(board_state, temperature=temp))
    while chosen_move not in board_state.get_legal():
        chosen_move = np.random.choice(8, p=tree.get_policy(board_state, temperature=temp))
    board_state.move(chosen_move)
    return chosen_move


def combo_engine(agent, tree, search_iters, search_threads, c_val, temp, game, previous_move):
    # print(game)
    # print("AB evaluating above board")
    ab_search = check_tactics(game.board, previous_move)   # first check any tactics for a win
    if ab_search[1] > 0 and ab_search[0] in game.get_legal():  # if a winning tactic detected and move is legal, play it
        game.move(ab_search[0])
        # print("ab decided tactic 1st ", ab_search)
        return ab_search[0]
    else:   # else, ask the nn to give a MCTS move, then try the move
        strategic_move = nn_search(agent, tree, search_iters, search_threads, c_val, temp, game, previous_move)
        # if NN gives same move suggestion as AB, it must be good (its tactically + positionally sound)
        if strategic_move != ab_search[0]:      # else, make sure the strategic NN move is tactically sound as well
            if check_tactics(game.board, strategic_move)[1] > 0:  # if find win after NN move itsa loss for combo engine
                game.unmove(strategic_move)    # since there was winning tactic after, its bad move, so undo it
                if ab_search[0] in game.get_legal():   # and play AB suggestion
                    game.move(ab_search[0])
                    # print("ab decided tactic 2nd ", ab_search[0])
                    return ab_search[0]
                else:
                    game.move(strategic_move)       # failsafe if AB suggestion is illegal by some bug in the code
                    # print(f"nn decided strategic cuz failsafe {ab_search[0]} illegal, so play ", strategic_move)
                    return strategic_move
            else:
                # print(f"nn decided strategic cuz no tactics ", strategic_move)
                return strategic_move
        else:
            # print(f"nn decided strategic nn and ab agreed ", strategic_move)
            return strategic_move    # NN and AB agreed on move, so it must be good


def test():
    """Run this (with c4_nn_agent.py as __main__) to test your nn's skills!"""
    # vs. NN, COMBO scored 72-3-25
    # vs. AB, COMBO scored
    combo_wins = 0
    enemy_wins = 0
    draws = 0
    games_num = 100
    # noinspection PyPep8Naming
    NN = Connect4NN("./connect4_models")  # load the most current version of the network

    for being_played in range(games_num):  # play 'games' num of games
        winner = "No one, yet"
        previous_move = 4  # dummy move needed for the poorly written AB engine
        game = cb.BoardState()  # generate a starting position
        combo_tree = mc.MCTS()
        switch = bool(random.getrandbits(1))  # pick a random player to go first
        temperature = 1.75
        for n in range(64):  # play out 64 moves
            if switch:
                combo_move = combo_engine(NN, combo_tree, 2400, 80, 4.0, temperature, game, previous_move)
                previous_move = combo_move   # above has side-effect of playing move on board
                if game.move_check(combo_move) == 1:  # if combo won, continue to the next game
                    combo_wins += 1
                    winner = "COMBO"
                    break
            else:
                # enemy_move = nn_search(NN, tree, 1200, 80, 4.0, temperature, game, previous_move)
                enemy_move = AB_move(game, previous_move)
                previous_move = enemy_move
                if game.move_check(enemy_move) == 1:  # if enemy agent wins
                    enemy_wins += 1
                    winner = "ENEMY"
                    break
            switch = not switch
            if n == 7:
                temperature = 1e-5
            if n == 63:  # draw
                winner = "DRAW"
                draws += 1
        print(being_played, winner)
    combo_score = 1.0 * combo_wins + 0.5 * draws
    enemy_score = games_num - combo_score
    print(f"Final score for COMBO: {combo_wins}-{draws}-{enemy_wins} ({combo_score}/{games_num})")
    print(f"Final score for ENEMY: {enemy_wins}-{draws}-{combo_wins} ({enemy_score}/{games_num})")


if __name__ == "__main__":
    test()
