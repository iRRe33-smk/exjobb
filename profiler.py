import pytorch_v2
def main():
    game = pytorch_v2.TorchGame(Horizon = 1, N_actions= 1, N_actions_startpoint= 1, I=5, D=2)
    game.Run()

def test():
    sum = 0
    for i in  range(10000):
        sum += (i % 5123) * 2
        sum = sum % 1236
    return f"test done, final value : {sum}"
if __name__ == "__main__":
    main()
