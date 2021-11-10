#include "stdafx.h"


void bitboard_parser(uint64_t* white, uint64_t* black, char* numpy_array)
{
	uint64_t pos = 63;
	for (char* it = numpy_array; *it; ++it)
	{
		if (*it == 'W') *white |= right1 << pos;
		else if (*it == 'B') *black |= right1 << pos;
		--pos;
	}
}



double time_diff(timespec one, timespec two){
	double nano = 1'000'000'000;
	double s_diff = nano * (double)(one.tv_sec - two.tv_sec);
	double n_diff = (double)(one.tv_nsec - two.tv_nsec);
	return (s_diff + n_diff) / nano;
}


int main(int argc, char **argv)
{
//	if (argc > 1)  // for the alpha zero competition, this converts from python to c++ and returns AB search move
//	{
//
//		uint64_t white_board = 0;
//		uint64_t black_board = 0;
//		Players playing_rn;
//		bitboard_parser(&white_board, &black_board, argv[1]);
//		if (argv[2][0] == 'W') playing_rn = Players::WHITE;
//		else playing_rn = Players::BLACK;
//		int prev_move = atoi(argv[3]);
//		BoardState game(&white_board, &black_board, playing_rn);
//		std::pair<int, int> result = AB_wrapper(game, 12, -32767, 32767, prev_move);
//		std::cout << result.first << " " << result.second;
//	}


	 uint64_t zero0 = 0;
	 uint64_t zero1 = 0;
	 BoardState game(&zero0, &zero1);

	 timespec start, end;
	 int pmove = 4;
	 std::pair<char, char> result;
	 int count = 0;
	 int static_depth;

	 if (argc == 2) static_depth = (int) atoi(argv[1]);
	 else if (argc <= 1) static_depth = 12;
	 else if (argc > 2) return -1;

	 std::cout << "Welcome to Connect 4!" << std::endl;
	 std::cout << "Today you will be playing against a 'super advanced' AI at Connect 4!" << std::endl;
	 std::cout << "Press the number keys from 0-7 to play in each column (ie. press 0 and then enter to play in the 0th (1st) column)" << std::endl;
	 std::cout << "Good luck (you will need it, the AI can be a little sadistic)!" << std::endl;


	 while (true)
	 {
	 	game.print();   // get player move
	 	std::cout << " |0|1|2|3|4|5|6|7|" << std::endl;
	 	while (true)
	 	{
	 		std::cout << "Input player move: ";
	 		std::cin.sync();
	 		std::string user_move;
	 		std::getline(std::cin, user_move);
	 		std::stringstream int_stream(user_move);
	 		if ((int_stream >> pmove) && user_move.length() == 1)
	 			if (!game.is_illegal(pmove)) break;
	 		std::cout << "Illegal move!" << std::endl;
	 	}
	      
	 	game.move(pmove);
	 	if (game.eval(pmove) == Players::WHITE) { std::cout << "Congrats you win!" << std::endl; break; }
	 	count++;
		clock_gettime(CLOCK_MONOTONIC, &start);
	 	result = (AB_wrapper(game, static_depth, -32767, 32767, pmove));  // returns (best_move, eval)
		clock_gettime(CLOCK_MONOTONIC, &end);

	 	game.move(result.first);
	 	if (game.eval(result.first) == Players::BLACK) { std::cout << "You lose!" << std::endl; break; }
	 	count++;

	 	// std::cout << "Move: " << (int)result.first << std::endl;
	 	// std::cout << "Eval: " << (int)result.second << std::endl;
	 	std::cout << "Took: " << time_diff(end, start) << " seconds" << std::endl;
	 	if (count >= 64) { std::cout << "Congrats you got a draw!" << std::endl;  break; }
	 }
	 game.print();

	return 0;
}


