import pytorch_v2
import cProfile
import re

if __name__ == "__main__":
    cProfile.run('re.compile("foo|bar")')
    FullGame = pytorch_v2.TorchGame(Horizon=3, N_actions=2, I=5, D=2)
    FullGame.Run()