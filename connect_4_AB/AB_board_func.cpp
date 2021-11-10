#include "stdafx.h"

void show_bin(uint64_t inpt)  // helper function for displaying 64 bit binary numbers in big-endian mode
{
	for (int i = 7; i != -1; i--)
	{
		std::cout << std::bitset<8>((((bit_mask << (8 * i)) & inpt)) >> (8 * i)) << '\'';
	}
	std::cout << std::endl;
}


BoardState::BoardState(uint64_t *white, uint64_t *black, Players whose_turn, int count) // default constructor
{
	WHITE_PIECES = *white; 
	BLACK_PIECES = *black;
	PLAYING = whose_turn;
	COUNT = count;
	//COUNT3 = count3;
}

void BoardState::print()
{
	std::string border = "******************";
	std::cout << border << std::endl;
	for (int i = 7; i != -1; --i)      // row iterator
	{
		std::cout << "|";
		for (int j = 0; j != 8; ++j)  // col iterator
		{
			std::cout << " ";
			if (WHITE_PIECES & ((static_cast<uint64_t>(1) << (63 - (i + 8 * j))))) std::cout << "W";
			else if (BLACK_PIECES & ((static_cast<uint64_t>(1) << (63 - (i + 8 * j))))) std::cout << "B";
			else std::cout << "-";
		}
		std::cout << "|" << std::endl;
	}
	std::cout << border << std::endl;
}



void BoardState::move(const unsigned char &move)  // efficient because this if statement has very predictable pattern (WBWBWBWBWBWBWBWBWBWBWB...)
{
	if (PLAYING == Players::WHITE)
	{
		WHITE_PIECES |= (right1 << (8 * (7 - move) + col_height[((WHITE_PIECES | BLACK_PIECES) & (cols[move])) >> (8 * (7 - move))]));
		PLAYING = Players::BLACK;
	}
	else
	{
		BLACK_PIECES |= (right1 << (8 * (7 - move) + col_height[((WHITE_PIECES | BLACK_PIECES) & (cols[move])) >> (8 * (7 - move))]));
		PLAYING = Players::WHITE;
	}
}

void BoardState::unmove(const unsigned char &move)
{
	if (PLAYING == Players::WHITE) // whites turn so black must have just played
	{
		BLACK_PIECES &= ~(right1 << (57 - 8 * move + col_height[((WHITE_PIECES | BLACK_PIECES) & (cols[move])) >> (8 * (7 - move))]));
		PLAYING = Players::BLACK;
	}
	else
	{
		WHITE_PIECES &= ~(right1 << (57 - 8 * move + col_height[((WHITE_PIECES | BLACK_PIECES) & (cols[move])) >> (8 * (7 - move))]));
		PLAYING = Players::WHITE;
	}
}

void BoardState::counts_move(const char &last_move, bool is_rev)
{
	short this_height = 6 - col_height[((WHITE_PIECES | BLACK_PIECES) & (cols[last_move])) >> (8 * (7 - last_move))];
	const char current_shift = 8 * last_move + this_height;
	const uint64_t current_piece = left1 >> (current_shift);
	char diff = 0;
	if (PLAYING == Players::BLACK)  // white just played so update count from white's perspective
	{
;
		// <ADDITIVE PARTS>
		//		<ROW PARTS>
		if (last_move != 7 && (WHITE_PIECES & (current_piece >> 8)))  // detected something to the right 1 col over 
		{
			if (0 <= last_move && last_move <= 4)  diff += (!(BLACK_PIECES & (row_rl >> (current_shift + 16)))) ? 1 : 0;		// check if 2 empty to the right of the group
			if (1 <= last_move && last_move <= 5)  diff += (!(BLACK_PIECES & (row_split >> (current_shift - 8)))) ? 1 : 0;     // check empty either side of group of piec
			if (2 <= last_move && last_move <= 6)  diff += (!(BLACK_PIECES & (row_rl >> (current_shift - 16)))) ? 1 : 0;       // check if 2 empty to the left of the group
		}

		if (last_move != 0 && (WHITE_PIECES & (current_piece << 8)))   // detected something to the left 1 col over
		{
			if (1 <= last_move && last_move <= 5) diff += (!(BLACK_PIECES & (row_rl >> (current_shift + 8)))) ? 1 : 0;			// right check allowed
			if (2 <= last_move && last_move <= 6)  diff += (!(BLACK_PIECES & (row_split >> (current_shift - 8)))) ? 1 : 0;     // split or "around" check allowed
			if (3 <= last_move && last_move <= 7)  diff += (!(BLACK_PIECES & (row_rl >> (current_shift - 24)))) ? 1 : 0;       // left check allowed
		}


		if (last_move != 0 && last_move != 1 && (WHITE_PIECES & (current_piece << 16)))  // deteced something to the left 2 col over
		{
			if (2 <= last_move && last_move <= 6) diff += (BLACK_PIECES & (row_half_split >> (current_shift - 8))) ? 0 : 1;   //     D_P_ check
			if (3 <= last_move && last_move <= 7) diff += (BLACK_PIECES & (row_half_split >> (current_shift - 24))) ? 0 : 1;   //   _D_P  check
		}

		if (last_move != 6 && last_move != 7 && (WHITE_PIECES & (current_piece >> 16)))  // deteced something to the right 2 col over
		{ 
			if (0 <= last_move && last_move <= 4) diff += (BLACK_PIECES & (row_half_split >> (current_shift + 8))) ? 0 : 1;         //     P_D_  check
			if (1 <= last_move && last_move <= 5) diff += (BLACK_PIECES & (row_half_split >> (current_shift - 8))) ? 0 : 1;        //    _P_D   check
		}
		//		</ROW PARTS>
		//		<COL PARTS>	
		if (1 <= this_height && this_height <= 6)   // these are nice and simple
		{
			diff += (WHITE_PIECES & (current_piece << 1)) ? 1 : 0;  // since it is known that the one above is empty and the current one is white, we only need to check the one below
		}
		//		</COL PARTS>
		//</ADDITIVE PARTS>
	}
	else //(PLAYING == Players::WHITE)  // white just played so update count from white's perspective
	{
		// <ADDITIVE PARTS>
		//		<ROW PARTS>
		if (last_move != 7 && (BLACK_PIECES & (current_piece >> 8)))  // detected something to the right 1 col over 
		{
			if (0 <= last_move && last_move <= 4)  diff += (!(WHITE_PIECES & (row_rl >> (current_shift + 16)))) ? -1 : 0;		// check if 2 empty to the right of the group
			if (1 <= last_move && last_move <= 5)  diff += (!(WHITE_PIECES & (row_split >> (current_shift - 8)))) ? -1 : 0;     // check empty either side of group of piec
			if (2 <= last_move && last_move <= 6)  diff += (!(WHITE_PIECES & (row_rl >> (current_shift - 16)))) ? -1 : 0;       // check if 2 empty to the left of the group
		}

		if (last_move != 0 && (BLACK_PIECES & (current_piece << 8)))   // detected something to the left 1 col over
		{
			if (1 <= last_move && last_move <= 5) diff += (!(WHITE_PIECES & (row_rl >> (current_shift + 8)))) ? -1 : 0;			// right check allowed
			if (2 <= last_move && last_move <= 6)  diff += (!(WHITE_PIECES & (row_split >> (current_shift - 8)))) ? -1 : 0;     // split or "around" check allowed
			if (3 <= last_move && last_move <= 7)  diff += (!(WHITE_PIECES & (row_rl >> (current_shift - 24)))) ? -1 : 0;       // left check allowed
		}


		if (last_move != 0 && last_move != 1 && (BLACK_PIECES & (current_piece << 16)))  // deteced something to the left 2 col over
		{
			if (2 <= last_move && last_move <= 6) diff += (WHITE_PIECES & (row_half_split >> (current_shift - 8))) ? 0 : -1;   //     D_P_ check
			if (3 <= last_move && last_move <= 7) diff += (WHITE_PIECES & (row_half_split >> (current_shift - 24))) ? 0 : -1;   //   _D_P  check
		}

		if (last_move != 6 && last_move != 7 && (BLACK_PIECES & (current_piece >> 16)))  // deteced something to the right 2 col over
		{
			if (0 <= last_move && last_move <= 4) diff += (WHITE_PIECES & (row_half_split >> (current_shift + 8))) ? 0 : -1;         //     P_D_  check
			if (1 <= last_move && last_move <= 5) diff += (WHITE_PIECES & (row_half_split >> (current_shift - 8))) ? 0 : -1;        //    _P_D   check
		}
		//		</ROW PARTS>
		//		<COL PARTS>	
		if (1 <= this_height && this_height <= 6)   // these are nice and simple
		{
			diff += (BLACK_PIECES & (current_piece << 1)) ? -1 : 0;  // since it is known that the one above is empty and the current one is white, we only need to check the one below
		}
		//		</COL PARTS>
		//</ADDITIVE PARTS>
	}
	COUNT += (is_rev) ? -diff : diff;
}


Players BoardState::eval(const char &last_move)  // sketchily serves 2 purposes, 1 to check for a win, 2 to update count2 and count3
{
	if (PLAYING == Players::BLACK)    // if black's turn, whitie just made a move and therefore may have won with his last move
	{
		short this_height = 6 - col_height[((WHITE_PIECES | BLACK_PIECES) & (cols[last_move])) >> (8 * (7 - last_move))];
		// std::cout << "(" << static_cast<int>(last_move) << ", " << this_height << ")" << std::endl;
		if (((row_test0 >> this_height) & WHITE_PIECES) == (row_test0 >> this_height)) return Players::WHITE;      // std::cout << "row0" << std::endl
		if (((row_test1 >> this_height) & WHITE_PIECES) == (row_test1 >> this_height)) return Players::WHITE;	  // std::cout << "row1" << std::endl
		if (((row_test2 >> this_height) & WHITE_PIECES) == (row_test2 >> this_height)) return Players::WHITE;	  // std::cout << "row2" << std::endl
		if (((row_test3 >> this_height) & WHITE_PIECES) == (row_test3 >> this_height)) return Players::WHITE;	  // std::cout << "row3" << std::endl
		if (((row_test4 >> this_height) & WHITE_PIECES) == (row_test4 >> this_height)) return Players::WHITE;	  // std::cout << "row4" << std::endl

		if (this_height >= 3)    // | check
		{
			uint64_t this_col_check = (col_test >> (8*last_move + this_height - 3));
			if ((WHITE_PIECES & this_col_check) == this_col_check) return Players::WHITE;
		}


		char hashed_val = (last_move << 3) + this_height;   // for diagonal checks

		char start_dl_ur = dl_ur_diag_idx[hashed_val][0];  // the down-left to up-right (/) checks
		char end_dl_ur = dl_ur_diag_idx[hashed_val][1];  // the down-left to up-right (/) checks
		for (int i = start_dl_ur; i <= end_dl_ur; i++)
		{
			uint64_t diag_check = (diag_test_dl_ur << (36 - (8 * last_move) - this_height + 9 * i));
			//show_bin(diag_check);
			if ((WHITE_PIECES & diag_check) == diag_check) return Players::WHITE;
		}

		char start_ul_dr = ul_dr_diag_idx[hashed_val][0];  // the up-left to down-right (\) checks
		char end_ul_dr = ul_dr_diag_idx[hashed_val][1];  // the up-left to down-right (\) checks
		for (int j = start_ul_dr; j <= end_ul_dr; j++)
		{
			uint64_t diag_check = diag_test_ul_dr << ((4 - this_height + 8 * (7 - last_move) - 7 * j));
			//show_bin(diag_check);
			if ((WHITE_PIECES & diag_check) == diag_check) return Players::WHITE;
		}
		return Players::NONE;
	}
	else   // PLAYING == PLAYERS::WHITE ( => it is white's turn, so black just moved which means black has a chance to have just won with his last move
	{
		char this_height = 6 - col_height[((WHITE_PIECES | BLACK_PIECES) & (cols[last_move])) >> (8 * (7 - last_move))];
		//std::cout << "(" << static_cast<int>(last_move) << ", " << this_height << ")" << std::endl;
		if (((row_test0 >> this_height) & BLACK_PIECES) == (row_test0 >> this_height)) return Players::BLACK; //  std::cout << "row0" << std::endl
		if (((row_test1 >> this_height) & BLACK_PIECES) == (row_test1 >> this_height)) return Players::BLACK; //	 std::cout << "row1" << std::endl
		if (((row_test2 >> this_height) & BLACK_PIECES) == (row_test2 >> this_height)) return Players::BLACK; //	 std::cout << "row2" << std::endl
		if (((row_test3 >> this_height) & BLACK_PIECES) == (row_test3 >> this_height)) return Players::BLACK; //	 std::cout << "row3" << std::endl
		if (((row_test4 >> this_height) & BLACK_PIECES) == (row_test4 >> this_height)) return Players::BLACK; //	 std::cout << "row4" << std::endl
		if (this_height >= 3)    // | check
		{
			uint64_t this_col_check = (col_test >> (8 * last_move + this_height - 3));
			if ((BLACK_PIECES & this_col_check) == this_col_check) return Players::BLACK;
		}


		char hashed_val = (last_move << 3) + this_height;   // for diagonal checks

		char start_dl_ur = dl_ur_diag_idx[hashed_val][0];  // the down-left to up-right (/) checks
		char end_dl_ur = dl_ur_diag_idx[hashed_val][1];  // the down-left to up-right (/) checks
		for (int i=start_dl_ur; i <= end_dl_ur; i++)
		{
			uint64_t diag_check = (diag_test_dl_ur << (36 - (8 * last_move) - this_height + 9*i));
			//show_bin(diag_check);
			if ((BLACK_PIECES & diag_check) == diag_check) return Players::BLACK;
		}

		char start_ul_dr = ul_dr_diag_idx[hashed_val][0];  // the up-left to down-right (\) checks
		char end_ul_dr = ul_dr_diag_idx[hashed_val][1];  // the up-left to down-right (\) checks
		for (int j=start_ul_dr; j <= end_ul_dr; j++)
		{
			uint64_t diag_check = diag_test_ul_dr << ((4 - this_height + 8 * (7 - last_move) - 7*j));
			//show_bin(diag_check);
			if ((BLACK_PIECES & diag_check) == diag_check) return Players::BLACK;
		}
		return Players::NONE;
	}
}


bool BoardState::is_illegal(const unsigned char &move)
{
	return ((WHITE_PIECES | BLACK_PIECES) & (left1 >> (8 * (move) + 7)));
}

