import math
import threading as th
import time
import numpy as np
import c4_nn_agent as c4_nn
import connect4_base as cb

"""Class for the Monte Carlo tree search, which uses a poorly implemented AlphaZero (by Google Deepmind) APV-MCTS 
algorithm, to generate an improved policy used for a training target and expected score for a given board state
(see AlphaZero paper for more details)"""


class MCTS(object):
    def __init__(self):
        # TREE STATISTICS AND PARAMETERS
        self.N = {}  # node counts
        self.P = {}  # prior distribution of moves
        self.Q = {}  # mean simulation rewards (includes virtual loss during search)
        self.W = {}  # total accumulated simulation rewards
        self.V = {}  # total virtual loss
        self.wins = {}   # hash table of all game_check's done so far, and their results
        self.visited = set()    # set of all visited nodes cuz set type is really fast to search through
        self.v_l = 3                # virtual loss amount

        # THREAD SYNCHRONIZATION LOCKS
        self.safety_locker = th.Lock()  # for accessing tree statistics (see thread_loop())
        self.num_locker = th.Lock()    # for accessing thread synchronization variables
        self.locker = th.Lock()        # for accessing the queue_eval function safely

        # THREAD SYNCHRONIZATION EVENTS
        self.eval_locker = th.Event()    # for determining when to do a gpu eval (when self.bucket is full)
        self.cont_locker = th.Event()  # continue lock, 'restart' search threads when this is set

        # THREAD SYNCHRONIZATION VARIABLES FOR BATCHED GPU EVAL (APV part of APV-MCTS)
        self.bucket = []            # gets filled with board eval requests which get sent to the gpu
        self.being_evaluated = set()  # set of nodes currently in gpu eval queue
        self.evals = []            # results from gpu eval are stored temporarily in here
        self.expect = 0       # defines the max number of boards evaluated at any time
        self.dont_need = 0  # number of threads that returned without needing a gpu eval
        self.got_eval = 0          # number of threads that have utilized their eval in a given bucket

    def reset(self, threads):
        """Reset thread synchronization variables to make sure the tree is ready for more simulations"""
        self.bucket = []
        self.expect = threads
        self.dont_need = 0
        self.being_evaluated.clear()
        self.got_eval = 0
        self.eval_locker.clear()
        self.cont_locker.clear()

    def search_iter(self, board_state, c4_nnet, last_played, alpha=[0.8] * 8, d=0.8, c_explore=4.0, threads=64, iters=1024, depr=0.96):
        """Main function that gets called when you want to do some simulations, and update tree statistics"""
        assert(iters % threads == 0)  # make sure you aren't missing and simulations from rounding errors
        self.reset(threads)  # reset all necessary fields to prepare for monte carlo simulations

        thread_list = []  # list of currently running threads: search threads and prediction thread (pred_thread)
        pred = th.Thread(target=self.pred_thread, args=(c4_nnet, board_state, ))
        pred.start()  # see pred_thread()
        thread_list.append(pred)

        for _ in range(threads):
            # search_threads that do the actual simulations
            s_thread = th.Thread(target=self.thread_loop,
                                 args=(cb.BoardState(white_black=board_state.board, board_heights=board_state.heights),
                                       last_played, c_explore, alpha, d, iters // threads, depr, ))
            s_thread.start()
            thread_list.append(s_thread)
        for thr in thread_list:
            thr.join()

    def pred_thread(self, nn_agent, dummy_board):
        """Thread that synchronizes all the search/simulation threads and actually gets the gpu eval and returns it"""
        while self.expect != 0:  # when all threads have completed every iteration, we no long expect any gpu evals
            cnt = 0
            while len(self.bucket) != self.expect - self.dont_need:  # wait for bucket to be filled
                time.sleep(1e-4)  # expect = total num of active search thread, dont_need = threads that have rolled out
                cnt += 1
                if cnt >= 30000:
                    print()
                    print('+'*75)
                    print(dummy_board)
                    print("pred")
                    print(self.expect)
                    print(self.dont_need)
                    print(len(self.bucket))
                    print(self.got_eval)
                    print(len(self.being_evaluated))
                    print('+' * 75)
                    break
            try:                   # to the same board as another thread (only 1 thread needs gpu eval)
                self.evals = nn_agent.bucket_predict(self.bucket)  # gpu eval bucket with nn
            except ValueError:   # if bucket is empty, no need to evaluate anything, but it raises ValueError
                pass
            with self.safety_locker, self.num_locker:
                save_size = len(self.bucket)  # number of threads that requested gpu eval
                self.got_eval = 0  # number of threads who've used requested gpu eval (all must use before moving on)
                self.dont_need = 0  # number of threads waiting on another thread to get gpu eval
                self.being_evaluated.clear()    # no threads are now being evaluated
                self.bucket = []  # empty bucket when gpu eval is done
                self.eval_locker.set()   # notify threads, eval has been completed, update tree stats with the gpu eval
            cnt = 0
            while self.got_eval != save_size:  # wait for all threads that needed a gpu eval to access it
                time.sleep(1e-4)
                cnt += 1
                if cnt >= 30000:
                    print()
                    print('+' * 75)
                    print(dummy_board)
                    print("eval")
                    print(self.expect)
                    print(self.dont_need)
                    print(len(self.bucket))
                    print(self.got_eval)
                    print(len(self.being_evaluated))
                    print('+' * 75)
                    break
            self.locker.acquire()   # block any threads from going into queue_eval() early
            self.eval_locker.clear()
            self.cont_locker.set()  # restart the next batch of search threads
            self.cont_locker.clear()
            self.locker.release()

    def thread_loop(self, board_state, last_played, c_explore, alpha, d, iters, depr):
        """For each simulation/search thread, they do 'iters' num of simulations and notify pred_thread when done"""
        for _ in range(iters):   
            self.thread_search(board_state, last_played, c_explore, alpha, d, depr)
        with self.num_locker:   # when a search thread has completed all simulations, tell pred_thread
            self.expect -= 1

    def thread_search(self, board_state, last_played, c_explore, alpha, d, depr):
        """a single simulation/rollout/backprop for a single thread"""
        b_hash = hash(board_state)   # hash the board_state so it can be uniquely identified for tree stats
        try:
            game_end = self.wins[b_hash]  # check to see if .move_check already computed (checks for win/draw/nothing)
        except KeyError:   # if win/draw/loss/nothing state of a board isnt found, compute it
            game_end = board_state.move_check(last_played)   # compute move_check if value not found
            self.wins[b_hash] = game_end   # add result to win table
        if game_end:  # returns 0.0 if nothing, 0.00000001 if draw, 1.0 if win
            return game_end * depr  # returns game_end * depr so that deep nodes dont affect root nodes too drastically

        v_maybe = self.queue_eval(b_hash, board_state, alpha, d, depr)   # see if node explored: if not, add it to gpu
        if v_maybe is not None:    # eval request bucket, if it is, just move on and expand further down
            return v_maybe         # if a gpu eval was requested, return its q score right now to update tree stats

        max_ucb, best_move = -float("inf"), -1.5   # -inf, some weird value (for debugging reasons)
        try:
            sum_sqrt = math.sqrt(sum(self.N[b_hash]))
        except KeyError:
            print("key error")
            print(self.expect)
            print(self.eval_locker.is_set())
            print(self.cont_locker.is_set())
            print(self.dont_need)
            print(len(self.bucket))
            print(len(self.being_evaluated))
            print(b_hash in self.being_evaluated)
            return 0
        for m in board_state.get_legal():  # computes UCB algorithm used in AlphaZero paper to pick a node for expansion
            ucb = self.Q[b_hash][m] + c_explore * (sum_sqrt / (1 + self.N[b_hash][m])) * self.P[b_hash][m]
            if ucb > max_ucb:
                max_ucb = ucb
                best_move = m
        with self.safety_locker:
            self.N[b_hash][best_move] += 1  # we are now about to explore the node so increase its visit count
            self.V[b_hash][best_move] += self.v_l   # add virtual loss to discourage threads from going down same path
            self.Q[b_hash][best_move] = (self.W[b_hash][best_move] - self.V[b_hash][best_move]) / self.N[b_hash][best_move]   # set q to be (total - virtual) / visits, discourages other threads to go down the same path ' blocking'
        board_state.move(best_move)  # having chosen a move with UCB, play that move and recursive call thread_search()
        v = self.thread_search(board_state, best_move, c_explore, alpha, d, depr)  # to expand further down
        board_state.unmove(best_move)   # using a move/unmove procedure instead of copying everything to save memory
        with self.safety_locker:  # after thread has returned we must update tree stats accordingly
            self.W[b_hash][best_move] += v   # increase total accumulated value
            self.V[b_hash][best_move] -= self.v_l   # take away virtual loss (encourage threads down same path again)
            self.Q[b_hash][best_move] = (self.W[b_hash][best_move] - self.V[b_hash][best_move]) / self.N[b_hash][best_move]    # set q to (total + virtual) / visits, 'unblocks' the thread
        return -v * depr    # return -v because (eg.) a +0.8 eval for white is losing at -0.8 for black the move before

    def queue_eval(self, b_hash, board_state, alpha, d, depr):
        """Checks to see if a node has already been visited/expanded, if not, adds it to gpu eval bucket,
        unless another thread has already done so"""
        self.locker.acquire()    # make sure we don't get duplicate threads requesting gpu eval for the same position
        if b_hash not in self.visited:   
            if b_hash not in self.being_evaluated:
                self.being_evaluated.add(b_hash)   # ok, this node hasn't been visited, and no other thread is
                with self.num_locker:               # currently requesting gpu eval of this position
                    queue_pos = len(self.bucket)  # locker so the order in which boards were added to bucket is reliable
                    self.bucket.append(board_state.board)   # so we can access the correct eval later
                self.locker.release()
                self.eval_locker.wait()  # wait for bucket to be filled and evaluated
                # self.eval_locker.clear()  # reset flag so next batch of search threads wait for gpu eval
                try:
                    v = self.evals[1][queue_pos]  # actually access the gpu eval
                except IndexError:      # because of tf.squeeze, if only 1 thing in self.bucket, this would index scalar
                    v = self.evals[1]   # dir(a), a < 1 => flatten pi, a > 1 => emphasize a random move
                self.P[b_hash] = d * self.evals[0][queue_pos] + (1 - d) * np.random.dirichlet(alpha)
                self.N[b_hash] = np.zeros(8).astype(np.int64)  # initialize N, Q, W, and V to zeros (expand)
                self.Q[b_hash] = np.zeros(8)
                self.W[b_hash] = np.zeros(8)
                self.V[b_hash] = np.zeros(8)
                self.visited.add(b_hash)
                with self.num_locker:
                    self.got_eval += 1      # after expansion, notify pred_thread that gpu eval has been used
                self.cont_locker.wait()  # synchronize with other threads (wait until threads have accessed gpu eval)
                # self.cont_locker.clear()
                return -v * depr   # return the q value from the gpu eval to update tree stats
            else:   # if not visited, but another thread has already requested gpu eval for this board state
                self.dont_need += 1  # this thread will not be requesting gpu eval, so tell pred_thread
                self.locker.release()  # save to add because we are already in a lock
                self.cont_locker.wait()  # wait for every thread below it to return its eval, proceed when we know
                # self.cont_locker.clear()  # that the eval needed will be there
        else:  # node already visited and eval is done, just move on
            self.locker.release()
        return None

    def get_policy(self, board_state, temperature=1.0):
        """Input is a BoardState object, returns normalized, exponentiated visit counts.
        as temp->0, policy->(some basis vector, more emphasized)  and as temp->1, policy->(regular distr, spread out)"""
        if self.N[hash(board_state)].sum() > 0:
            if temperature >= 0.3:   # returns a MCTS improved policy based on root node visit counts
                exponentiated = np.copy(self.N[hash(board_state)]).astype(np.float) ** (1. / temperature)
                return exponentiated / exponentiated.sum()
            else:   # for infinitesimal temperatures
                zero_array = np.zeros(8).astype(np.float64)
                zero_array[int(np.argmax(self.N[hash(board_state)]))] = 1.
                return zero_array
        else:
            return [0.125] * 8

    def get_eval(self, board_state):
        """returns the total score propogated upward (W) divided by the total number of visits"""
        total_sum = self.N[hash(board_state)].sum()
        if total_sum > 0:
            return np.divide(self.W[hash(board_state)].sum(), total_sum)
        else:
            return 0


def aux_func(an_board, some_tree, an_agent, lm):
    some_tree.search_iter(an_board, an_agent, lm, iters=1600, threads=100)


def test():
    """Run MCTS as main to test if everything is working, it takes about 1.2s on my machine, and should say that move 5 is best, and 
    the eval should be a pretty large positive"""
    an_board = cb.BoardState()
    some_tree = MCTS()
    an_agent = c4_nn.Connect4NN("./connect4_models")  # load board position based on its string appearance, its obvious that move 5 is best
    board_str = """**************************  
                   | -  -  -  -  -  -  -  - |
                   | -  -  -  -  -  -  -  - |
                   | -  -  -  -  -  -  -  - |
                   | -  -  -  -  -  -  -  - |
                   | -  -  -  -  -  -  -  - |
                   | -  -  -  -  -  W  -  - |
                   | W  -  B  -  -  W  -  - |
                   | B  -  B  -  B  W  W  - |
                   **************************"""
    an_board.to_board(board_str)   # of course assuming that you are playing white (which this implementation always does)
    an_agent.predict(an_board)
    print(an_board)
    start = time.time()
    aux_func(an_board, some_tree, an_agent, 2)
    print(f"took {time.time()-start}")
    print("tree size:", len(some_tree.visited))
    print("distr = ", list(some_tree.get_policy(an_board)))
    print("eval: ", some_tree.get_eval(an_board))
    print("q:", list(some_tree.Q[hash(an_board)]))
    print("w:", list(some_tree.W[hash(an_board)]))
    print("p:", list(some_tree.P[hash(an_board)]))
    print("n:", list(some_tree.N[hash(an_board)]))


if __name__ == "__main__":
    test()
