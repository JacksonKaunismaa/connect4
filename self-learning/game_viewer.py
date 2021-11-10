import sys
import pickle
import os


def view_game(file_path):
    try:
        with open(file_path, "rb") as f:
            game = pickle.load(f)
    except FileNotFoundError:
        with open(os.path.join("c4_games", file_path), "rb") as f:
            game = pickle.load(f)
    switch = True
    for game_state in game:
        if switch:
            print(game_state[0].reverse())  # flip the board and print the opposite eval
            try:
                print(-game_state[2])
            except IndexError:
                pass
            except TypeError:
                print(game_state[2])
        else:
            print(game_state[0])
            try:
                print(game_state[2])
            except IndexError:
                pass
        switch = not switch
        print(list(game_state[1]))



if __name__ == "__main__" and len(sys.argv) > 1:
    view_game(sys.argv[1])
else:
    print("Please specify a file name")
