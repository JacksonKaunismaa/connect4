CXX = g++

CXXFLAGS = -O3

SRC_FILES = AB_board_func.cpp  AB_search.cpp  connect_4_AB.cpp
COMBO_ENGINE_FLAG = -DCOMBO_ENGINE

playable: AB_board_func.cpp  AB_search.cpp  connect_4_AB.cpp
	$(CXX) -o c4_AB_playable $(SRC_FILES) $(CXXFLAGS)

combo: AB_board_func.cpp  AB_search.cpp  connect_4_AB.cpp
	$(CXX) -o c4_AB_combo $(SRC_FILES) $(CXXFLAGS) $(COMBO_ENGINE_FLAG)



