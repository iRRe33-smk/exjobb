from main import TorchGame
from matplotlib import pyplot as plt
import numpy as np
if __name__ == "__main__":
    # FullGame = TorchGame(Horizon=3, Max_actions_chosen=4, N_actions_startpoint=6, I=5, D=2,
    #                      Stochastic_state_update=False, Start_action_length=[10, 10], Max_optim_iter=5)

    # print(FullGame.techToParams(FullGame.InitialState))
    # hist = FullGame.Run()
    #hist.save_to_file("smallRun.pkl")
    # hist.send_email(test=False)


    game =  TorchGame(Horizon=3, Max_actions_chosen=4, N_actions_startpoint=6, I=5, D=2,
                         Stochastic_state_update=False, Players_action_length=[10, 10], Max_optim_iter=5, base_params="custom")


    scores = []
    # irange = range(1,1000,10)
    irange = np.int32(np.geomspace(1,1024,100))
    for i in irange:
        st = game.InitialState
        theta_0 = game.techToParams(st)
        score = sum([game.SalvoBattleSequential(theta=theta_0) for _ in range(i)])/i
        scores.append(score)

    plt.plot(irange, scores)
    plt.title("objective function with different number of draws")
    plt.ylabel("average score")
    plt.ylim(0,1)
    plt.xlabel("number of draws")
    plt.xscale("log")
    plt.show()
    plt.savefig("figures/objective_smoothness.png")
