#!/usr/bin/env python
import argparse
import c4_nn_agent as c4_nn
import connect_4_A0 as train


def main():
    parser = argparse.ArgumentParser(description='Connect4')
    parser.add_argument('--mode', type=str, default='play',
                        choices=['play', 'train'],
                        help='play or train')
    parser.add_argument('--opponent', type=str, default='NN',
                        choices=['NN', 'NN_AB', 'AB'],
                        help='Opponent type')
    args = parser.parse_args()

    agent = c4_nn.Connect4NN('connect4_models', log_path='/tmp/tensorflow_logs/c4_logs')
    if args.mode == 'play':
        if args.opponent == 'NN':
            agent.play_human()
        elif args.opponent == 'NN_AB':
            agent.play_combo()
        elif args.opponent == 'AB':
            agent.play_AB()
    elif args.mode == 'train':
        train.main(proc_num=8)


if __name__ == '__main__':
    main()