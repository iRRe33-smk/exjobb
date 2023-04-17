from pytorch_v2 import TorchGame
if __name__ == "__main__":
    FullGame = TorchGame(Horizon=3, Max_actions_chosen=4, N_actions_startpoint=6, I=5, D=2,
                         Stochastic_state_update=False, Start_action_length=[10, 10], Max_optim_iter=5)

    # print(FullGame.techToParams(FullGame.InitialState))
    hist = FullGame.Run()
    #hist.save_to_file("smallRun.pkl")
    # hist.send_email(test=False)