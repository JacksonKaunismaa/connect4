#pragma once
#include "precomputed_constants.h"
class BoardState // 8x8 connect 4 board
{
	public: // public (access specifier) members => can be accessed outside of the class (default is private => can only be accessed by other members of the class
		uint64_t WHITE_PIECES;    // 8 bytes, one for each column (eg. the 2nd bit represents the 2nd hole in the 1st column, the 11th bit is the 3rd hole in the 2nd column)
		uint64_t BLACK_PIECES;   // see stockfish board representation "bitboards"
		Players PLAYING;
		int COUNT;  // # of 2 in a row streaks that white has - ones black has
		//char COUNT3;  // # of 3 in a row streaks that white has - ones black has
		BoardState(uint64_t *white = 0, uint64_t *black = 0, Players whose_turn = Players::WHITE, int count = 0);  // default constuctor (note same name as class, and no return type)
	void print();  // representation of the current board state using W and B characters as the pieces
	bool is_illegal(const unsigned char &move);
	void move(const unsigned char &move);   // given an index [0, 7] adds a piece to the board in that column (switching between piece types done automatically)
	void unmove(const unsigned char &move);  // undo last move
	Players eval(const char& last_move);    // given the last move that was played, it returns whether Black, White, or None (no win so far) won with their last move
	void counts_move(const char &last_move, bool is_rev = false);  //   update COUNT2 and COUNT3 when a move is made

};

int alpha_beta_search(BoardState &board_state, char depth, int alpha, int beta, char last);   // returns an eval
std::pair<char, int> AB_wrapper(BoardState &board_state, char depth, int alpha, int beta, char last);  // returns (best_move, eval)
int eval_func(BoardState&, const Players &terminality, const char &last_move);   // returns static eval for given position
void update_wb_moves(char last, bool white_to_move);   // update the order in which moves are considered
void show_bin(uint64_t bit_board);

