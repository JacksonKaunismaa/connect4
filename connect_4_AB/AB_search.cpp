#include "stdafx.h"

int alpha_beta_search(BoardState &board_state, char depth, int alpha, int beta, char last)   // returns an eval
{ 
	Players is_terminal = board_state.eval(last);
	if ((depth == 0) || (is_terminal != Players::NONE))
	{
		return eval_func(board_state, is_terminal, last);
	}
	if (board_state.PLAYING == Players::WHITE)    // maximizing players turn
	{
		for (unsigned char m : white_moves)
		{
			if (!board_state.is_illegal(m))
			{
				board_state.move(m);
				//board_state.counts_move(m);  // update "streak" counter 
				int new_val = alpha_beta_search(board_state, depth - 1, alpha, beta, m);
				//board_state.counts_move(m, true);   // unupdate "streak" counter (symmetric function! ie. if f(x) = y, f(y) = x)
				board_state.unmove(m);
				alpha = (new_val > alpha) ? new_val : alpha;       // max of value and alpha
				if (alpha >= beta) return alpha;
			}
		}
		return alpha;
	}
	else  // Black (minimizing player's turn)
	{
		for (unsigned char m : black_moves)
		{
			if (!board_state.is_illegal(m))
			{
				board_state.move(m);
				//board_state.counts_move(m);
				int new_val = alpha_beta_search(board_state, depth - 1, alpha, beta, m);
				//board_state.counts_move(m, true);
				board_state.unmove(m);
				beta = (new_val < beta) ? new_val : beta;       // the lower of beta and value
				if (beta <= alpha) return beta;
			}
		}
		return beta;
	}
}

std::pair<char, int> AB_wrapper(BoardState &board_state, char depth, int alpha, int beta, char last) // assuming depth != 0 (returns best_move, eval)
{
	char best_move = 4;
	if (board_state.PLAYING == Players::WHITE)    // maximizing players turn
	{
		update_wb_moves(last, true);
		for (unsigned char m : white_moves)
		{
			if (!board_state.is_illegal(m))
			{
				board_state.move(m);
				//board_state.counts_move(m);
				int new_val = alpha_beta_search(board_state, depth - 1, alpha, beta, m);
				//board_state.counts_move(m, true);
				board_state.unmove(m);
				if (new_val > alpha)
				{
					alpha = new_val;
					best_move = m;
				}
				if (alpha >= beta) return std::pair<char, int>(best_move, alpha);
			}
		}
		return std::pair<char, int>(best_move, alpha);
	}
	else  // Black (minimizing player's turn)
	{
		update_wb_moves(last, false);
		for (unsigned char m : black_moves)
		{
			if (!board_state.is_illegal(m))
			{
				board_state.move(m);
				//board_state.counts_move(m);
				int new_val = alpha_beta_search(board_state, depth - 1, alpha, beta, m);
				//board_state.counts_move(m, true);
				board_state.unmove(m);
				if (new_val < beta)
				{
					beta = new_val;
					best_move = m;
				}
				beta = (new_val < beta) ? new_val : beta;       // the lower of beta and value
				if (beta <= alpha) return std::pair<char, int>(best_move, beta);
			}
		}
		return std::pair<char, int>(best_move, beta);
	}
}

int eval_func(BoardState &board_state, const Players &terminality, const char &move)
{
	return (int)terminality;
	//if (terminality != Players::NONE) return (int)terminality;
	//else return board_state.COUNT;
}

void update_wb_moves(char last, bool white_to_move)  // changes move order to be centered around the last played move, to look for blocks and stuff
{
	if (white_to_move) {
		std::swap(white_moves[0], white_moves[std::distance(std::begin(white_moves), std::find(std::begin(white_moves), std::end(white_moves), last))]);
		if (0 < last && last < 7) {  // fancy one liner to swap in place the position of white_moves[1] (2nd slot) and wherever last - 1 or last + 1 is
			std::swap(white_moves[1], white_moves[std::distance(std::begin(white_moves), std::find(std::begin(white_moves), std::end(white_moves), last - 1))]);
			std::swap(white_moves[2], white_moves[std::distance(std::begin(white_moves), std::find(std::begin(white_moves), std::end(white_moves), last + 1))]);
		}
		else if (last == 7) std::swap(white_moves[1], white_moves[std::distance(std::begin(white_moves), std::find(std::begin(white_moves), std::end(white_moves), last - 1))]);
		else if (last == 0) std::swap(white_moves[1], white_moves[std::distance(std::begin(white_moves), std::find(std::begin(white_moves), std::end(white_moves), last + 1))]);
	}
	else {
		std::swap(black_moves[0], black_moves[std::distance(std::begin(black_moves), std::find(std::begin(black_moves), std::end(black_moves), last))]);
		if (0 < last && last < 7) { 
			std::swap(black_moves[1], black_moves[std::distance(std::begin(black_moves), std::find(std::begin(black_moves), std::end(black_moves), last - 1))]);
			std::swap(black_moves[2], black_moves[std::distance(std::begin(black_moves), std::find(std::begin(black_moves), std::end(black_moves), last + 1))]);
		}
		else if (last == 7) std::swap(black_moves[1], black_moves[std::distance(std::begin(black_moves), std::find(std::begin(black_moves), std::end(black_moves), last - 1))]);
		else if (last == 0) std::swap(black_moves[1], black_moves[std::distance(std::begin(black_moves), std::find(std::begin(black_moves), std::end(black_moves), last + 1))]);
	}
}

