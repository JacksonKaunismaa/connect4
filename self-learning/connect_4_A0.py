import multiprocessing as mp
import os
import pickle
import random
import subprocess
import time
from os.path import join
import numpy as np
import c4_nn_agent as c4_nn
import connect4_base as cb
import monte_carlo as mc
import sys

"""Main file through which the training loop runs, including self play games, comparison against past iterations, comparison to an
objective alpha beta (AB) engine (using fixed depth of 13 ply), and all necessary IO operations."""


def play_AB(games, proc_id):
    """Compare the strength of the self-learning neural network agent to a traditional AB engine by playing some games"""
    nn_wins = 0
    ab_wins = 0
    draws = 0
    NN = c4_nn.Connect4NN("./connect4_models", device=proc_id)  # load the most current version of the network

    for being_played in range(games):  # play 'games' num of games
        winner = "No one, yet"
        previous_move = 4  # dummy move needed for the poorly written AB engine
        game_data = []  # save game data to watch them later
#        print(being_played)  # to show progress
        game = cb.BoardState()  # generate a starting position
        tree = mc.MCTS()  # init game tree for neural network (NN)
        switch = bool(random.getrandbits(1))  # pick a random player to go first
        temperature = 1.75
        if switch:
            order = "AB-NN"
        else:
            order = "NN-AB"
        for n in range(64):  # play out 64 moves
            if switch:
                output = subprocess.run(  # get AB move
                    ["./connect_4_AB.exe", cb.parse_board(game.board), "W", str(previous_move)],
                    stdout=subprocess.PIPE)
                numeric_move = int(output.stdout.decode("utf-8"))  # get AB move
                if numeric_move in game.get_legal():  # make sure the AB move is legal, otherwise play a random legal move
                    game.move(numeric_move)
                else:
                    game.move(random.choice(game.get_legal()))
                if game.move_check(numeric_move) == 1:  # if AB won, continue to the next game
                    ab_wins += 1
                    winner = "AB"
                    break
            else:
                result, previous_move = play_turn(NN, tree, 1600, 80, 4.0, temperature, game, previous_move)
                if result:  # if nn agent wins
                    nn_wins += 1
                    winner = "NN"
                    break
            switch = not switch
            try:  # try to get an idea of what moves the neural network was thinking about
                game_data.append((cb.BoardState(game.board), tree.get_policy(game), tree.get_eval(game)))
            except KeyError:
                game_data.append((cb.BoardState(game.board), "AB played an unexpected move!", "Eval unknown!"))
            if n == 7:
                temperature = 1e-5
            if n == 63:  # draw
                winner = "DR"
                draws += 1

        game_name = join("c4_games", f"{order}(v{NN.get_step()})-game-{int(time.time() * 100.)}-{winner}.pickle")
        with open(game_name, "wb") as f:  # save the game
            pickle.dump(game_data, f)
    nn_score = 1.0 * nn_wins + 0.5 * draws
    ab_score = games - nn_score
    print(f"Final score for NN: {nn_wins}-{draws}-{ab_wins} ({nn_score}/{games})")
    print(f"Final score for AB: {ab_wins}-{draws}-{nn_wins} ({ab_score}/{games})")
    NN.kill()


def add_rewards(train_game, gamma=0.97):
    """+ number if win, - number if lost, 0 if draw, assigning rewards so nn learns to predict q properly"""
    reward = 1.
    for sample in train_game[::-1]:
        sample[2] += reward / 2  # divide by 2 so its an average of improved q and actual score z
        reward *= -gamma


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


def play_turn(agent, tree, search_iters, search_threads, c_val, temp, board_state, prev_move):
    """Given a board position, nn agent, search tree, and simulation parameters, it plays a move on the board and returns a tuple of
    (game_result, move_played) where game_result = 1 if the move was a win, else 0"""
    tree.search_iter(board_state, agent, prev_move, iters=search_iters, threads=search_threads, c_explore=c_val)
    chosen_move = np.random.choice(8, p=tree.get_policy(board_state, temperature=temp))
    while chosen_move not in board_state.get_legal():
        chosen_move = np.random.choice(8, p=tree.get_policy(board_state, temperature=temp))
    board_state.move(chosen_move)
    if board_state.move_check(chosen_move) == 1:
        return 1, chosen_move  # 1 corresponds to a win being detected
    return 0, chosen_move  # 0 corresponds to nothing being detected


def compete_one_game(old, new, search_iters, search_threads, c_val):
    """To compare the newly trained nn to the old version, we pit them against each other and see who wins"""
    old_tree = mc.MCTS()  # old nn and new nn need to have separate trees because otherwise the old nn might get
    new_tree = mc.MCTS()  # extra insight from the new version it might not have realized
    new_game = cb.BoardState()  # generate a new board and pick a random player to go first
    switch = bool(random.getrandbits(1))
    started = "with new playing as white,"  # display message so we can see if the new version is winning equal numbers of games
    if switch:  # as both white and black (was an issue early in development)
        started = "with new playing as black,"
    current_temp = 1.75  # temperature starts at 1.75 for the first 7 moves (exploration) then the rest use an infinitesimal amount
    prev_move = 0  # dummy starting move for the monte carlo search
    for dummy in range(64):
        if switch:  # if old nn's turn
            result, prev_move = play_turn(old, old_tree, search_iters, search_threads, c_val, current_temp, new_game,
                                          prev_move)
            if result:
                print(started, "old one won one")
                return -1
        else:  # if new nn's turn
            result, prev_move = play_turn(new, new_tree, search_iters, search_threads, c_val, current_temp, new_game,
                                          prev_move)
            if result:
                print(started, "new won!")
                return 1  # 1 corresponds to the new nn winning the game
        switch = not switch
        if dummy == 7:  # after 1st 7 exploratory moves, only only choose best ones by using "infinitesimal" temperature
            current_temp = 1e-5
    print(started, "there was a draw?!")
    return 0


def compete(old, new, c_val, games=51, search_iters=1000, search_threads=50):
    """Compares the old and new models to see if the new one really learned anything"""
    old_score = 0.
    new_score = 0.
    for _ in range(games):
        game_result = compete_one_game(old, new, search_iters, search_threads, c_val)
        if game_result == -1:
            old_score += 1.
        elif game_result == 1:
            new_score += 1.
        else:
            old_score += 0.5
            new_score += 0.5
    return float(new_score) / games


def try_training(games_num, c_val, p_id=0, perf_list=None):
    """After training data through self play has been created, train the nn on it to see if any improvements can be made
    after the nn has been actually trained on the data, compare this 'new' version to the old, untrained version to make sure we
    are actually learning the right things"""
    old_model = c4_nn.Connect4NN("./connect4_models", device=p_id)
    new_model = c4_nn.Connect4NN("./connect4_models",
                                 log_path="/tmp/tensorflow_logs/c4_new_head2",
                                 device=p_id)  # log some training info stuff
    training_data = []  # get training data from past game generation iters
    games_loc = os.fsencode(
        join("datasetts", "c4_train_games"))  # load all training data files in the c4_train_games folder
    for file in os.listdir(games_loc):
        filename = os.fsdecode(file)
        with open(join("datasetts", "c4_train_games", filename), "rb") as f:
            training_data += list(pickle.load(f))

    training_data = deduplicate(training_data)  # deduplicate and shuffle data
    random.shuffle(training_data)
    new_model.train_update(training_data,
                           epochs=2)  # train the new nn on the data, running over it 3 times for good measure
    print("Done training model, comparing trained new model to old model...")
    new_performance = compete(old_model, new_model, c_val, games=games_num)  # compare the trained and old version
    print("Done comparing to old model..")
    if perf_list is None:
        if new_performance >= 0.55:  # if the new version wins more than 55% of the games (assuming they are exactly equal, this happens over 51
            print(
                f"New model won by {new_performance}!")  # games about 7% of the time by pure chance), then it becomes the new model and is saved
            new_model.save()
            new_model.kill()
            old_model.kill()
        else:  # however if new made no significant improvement (didn't win more than 55%) then just keep the old version
            print(f"Old model won by {1. - new_performance}")
            old_model.save()
            new_model.kill()
            old_model.kill()
    else:
        perf_list.append(new_performance)
        if p_id == 0:
            while len(perf_list) != proc_num:
                time.sleep(0.08)
            new_performance = sum(perf_list) / proc_num
            if new_performance >= 0.55:  # if the new version wins more than 55% of the games (assuming they are exactly equal, this happens over 51
                print(
                    f"New model won by {new_performance}!")  # games about 7% of the time by pure chance), then it becomes the new model and is saved
                new_model.save()
                new_model.kill()
                old_model.kill()
            else:  # however if new made no significant improvement (didn't win more than 55%) then just keep the old version
                print(f"Old model won by {1. - new_performance}")
                old_model.save()
                new_model.kill()
                old_model.kill()


def generate(games_num, window, c_val, proc_id=0):
    """Does one iteration of game generation (plays games_num games to generate training data), trains new_model on that
     and then if it does better than old_model, it new_model is saved, else old_model is saved"""
    agent = c4_nn.Connect4NN('./connect4_models', device=proc_id)  # load self play agent
    global_training_data = []  # store training data
    global_tree = mc.MCTS()  # tree that gets reused between games to improve simulations
    print("Getting training data...")
    pct = 0.04
    pct_2 = 0.1
    for _ in range(games_num):
        if pct <= float(_) / games_num:  # every 4% of 2048 ~= 82 games, save current training data
            print("#", end='', flush=True)
            pct += 0.04
            with open(join(f"train_progress", f"train_dump_{proc_id}_-{int(pct*1000)}.pickle"), "wb") as f:
                pickle.dump(global_training_data, f)
        if pct_2 <= float(_) / games_num:
            global_tree = mc.MCTS()  # emtpy out global tree, to save memory every so often and also given an indicator of progress
            pct_2 += 0.1
            print("|", end='', flush=True)
        play_game(agent, global_tree, global_training_data,
                  c_val)  # actually generate the training data through self play
    agent.save()  # save the current agent
    agent.kill()  # free some memory
    window_size = 10
    try:  # relocate old trainng data so it isn't deleted
        with open(join("datasetts", "c4_train_games", f"training_data_{proc_id}_-{window % window_size}.pickle"), "rb") as f:
            move_this = pickle.load(f)
        with open(join("datasetts", "old_c4_games", f"training_data_{proc_id}_-{int(time.time()*1000)}.pickle"), "wb") as f:
            pickle.dump(move_this, f)
    except FileNotFoundError:
        pass
    with open(join("datasetts", "c4_train_games", f"training_data_{proc_id}_-{window % window_size}.pickle"), "wb") as f:
        pickle.dump(global_training_data, f)
    print("\nDone getting training data, training model..")


def play_game(agent, global_tree, global_training, c_val, search_iters=1000, search_threads=50):
    """Generates 1 game's worth of self play training data with samples of the form [state (s), improved policy (pi), reward (z)]"""
    game = cb.BoardState()
    training = []  # training data for this particular game
    chosen_move = 0
    for dummy in range(64):  # max number of turns per game
        global_tree.search_iter(game, agent, chosen_move, iters=search_iters, threads=search_threads)
        training.append(np.array([np.copy(game.board), global_tree.get_policy(game), global_tree.get_eval(game) / 2]))
        chosen_move = np.random.choice(8, p=global_tree.get_policy(game))
        while chosen_move not in game.get_legal():  # in theory this shouldn't be necessary
            chosen_move = np.random.choice(8, p=global_tree.get_policy(game))
        game.move(chosen_move)
        if game.move_check(chosen_move) == 1:
            add_rewards(training)
            global_training += training
            return
    global_training += training


def main():
    my_c = 4.0
    for cnt in range(40):
        if proc_num > 1:
            pool = mp.Pool(proc_num)
            manager = mp.Manager()

            args = [(1024, cnt, my_c, i) for i in range(proc_num)]
            pool.starmap(generate, args)
            pool.close()
            pool.join()

            pool = mp.Pool(proc_num)
            perf_list = manager.list()
            args = [(128//proc_num, my_c, i, perf_list) for i in range(proc_num)]
            pool.starmap(try_training, args)
            pool.close()
            pool.join()

            pool = mp.Pool(proc_num)
            args = [(25, i) for i in range(proc_num)]
            pool.map(play_AB, [25]*proc_num)
            pool.close()
            pool.join()
        else:
            p = mp.Process(target=try_training, args=(21, my_c,))  # try to train neural net on the games just played out
            p.start()
            p.join()  # its a process to manage memory efficiently (so gpu memory can be cleared)
            p = mp.Process(target=play_AB, args=(3,))  # play 25 game matches against AB engine
            p.start()
            p.join()
            p = mp.Process(target=generate, args=(100, cnt, my_c,))  # do 768 games of self play
            p.start()
            p.join()
            print(f"Total sessions completed: {cnt}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        proc_num = int(sys.argv[1])
    else:
        proc_num = 1
    main()
