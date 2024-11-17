# Connect 4 reinforcement learning agent

The first component, the self-learning module, uses deep reinforcement learning to train a residual CNN to play connect 4 on an 8x8 board.

The second component, the connect_4_AB module, uses an alpha-beta search, bit boards, and a simple evaluation function (win/loss).


# Usage


### Build AB search engine (playable by human)

```bash
cd connect_4_AB
make playable
./c4_AB_playable
```

### Build AB search engine for use with the combo (AB + NN) engine
```bash
cd connect_4_AB
make combo
cp c4_AB_combo ../self-learning
```

### To train the NN:
```bash
cd self-learning
./main.py --mode train
```

### To play against the engine:
You can access trained weights from the release at https://github.com/JacksonKaunismaa/connect4/releases/tag/v1.0
```bash
cd self-learning
./main.py --mode play --opponent {NN, NN_AB, AB}
```