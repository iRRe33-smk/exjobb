import pandas
import torch
# from torch.autograd import grad, Function,functional
import numpy as np
import time
import json
import pandas as pd

def PCA(X: torch.Tensor):
    covs = torch.cov(X)
    covs.type(torch.complex64)

    vals, vec = torch.linalg.eig(covs)

    explained_variance = torch.cumsum(
        torch.abs(vals), -1) / torch.sum(torch.abs(vals))

    comps = vec @ covs

    return vals, vec, explained_variance, comps


class TorchGame():
    def __init__(self, Horizon=5, N_actions=5, N_actions_startpoint=100, Start_action_length=[10, 10], I=1,
                 D=3) -> None:
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        # torch.manual_seed(1337)
        # global variables
        #self.N_Technologies = N_Technologies
        self.Horizon = Horizon
        self.N_actions_startpoint = N_actions_startpoint
        self.N_actions = N_actions
        self.Plyers_action_length = torch.tensor(Start_action_length)
        # Used in TRL calculations
        self.I = I
        self.D = D


        self.FINAL_ACTIONS = []
        self.History = []
        self.Q = []


        df_stat = pd.read_excel("config_files/State_Conversion.xlsx", sheet_name="StartingState", header=0, index_col=0)
        print(df_stat)
        self.InitialState = torch.tensor(df_stat.astype(float).values)

        df_capaMat = pd.read_excel("config_files/State_Conversion.xlsx", sheet_name="ConversionMatrix", header=0, index_col=0)
        self.PARAMCONVERSIONMATRIX = torch.tensor(df_capaMat.astype(float).values)
        print(df_capaMat)


        self.N_Capabilities,  self.N_Technologies = self.PARAMCONVERSIONMATRIX.size()

        with open("config_files/xi_params.json") as f:
            params = json.load(f)
            self.xi_params_mu = torch.tensor(params["mu"][:self.N_Technologies])
            self.xi_params_sigma = torch.tensor(params["sigma"][:self.N_Technologies])



    def _randomPointsInSphere(self,rad=1):
        #https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
        
        nPoints = self.N_actions_startpoint
        nDim = self.N_Technologies
        params = (torch.rand(size = (nPoints,nDim+1)) + 1) / 2
        #radius [0,1] * r
        #angles [0,pi/2]
        
        r = params[:,-1] * rad
        phi = params[:,:-1] * torch.pi / 2
        
        
        X = torch.ones(nPoints,nDim) 
        for i in range(nDim-1):
            X[:,i] *= torch.cos(phi[:,i])
            #print(f"c{i}")
            for j in range(i):
                X[:, i] *= torch.sin(phi[:, j])
                # print(f"s{j}")

        for j in range(nDim):
            X[:, nDim - 1] *= torch.sin(phi[:, j])
            # print(f"s{j}")

        X = (r * torch.eye(nPoints)) @ X  # stretching by radius
        return X

    def Update_State(self, State, Action):

        xi_1 = torch.normal(mean=self.xi_params_mu, std=self.xi_params_sigma)
        xi_2 = torch.normal(mean=self.xi_params_mu, std=self.xi_params_sigma)
        xi = torch.exp(torch.stack((xi_1, xi_2), dim=-1))
        UpdateValue = Action * xi

        newState = torch.add(State, UpdateValue)

        return newState

    def TechnologyReadiness(self,State):
        
        trl = torch.pow(1+torch.exp(-State*(1/self.I)+self.D),-1)

        
        return trl

    def techToParams(self, State):

        trl = self.TechnologyReadiness(State)
        
        theta = torch.matmul(self.PARAMCONVERSIONMATRIX, trl)

        return theta

    def Battle(self, theta):
        results = torch.sum(theta, dim=1) / torch.sum(theta)
        return results

    def InitiativeProbabilities(self, phi, psi, sigma=1, c=1.6):

        stdev = 1.4142 * sigma
        dist = torch.distributions.Normal(loc=phi - psi, scale=stdev)

        critval = torch.tensor(c * stdev)

        p1_favoured = 1 - dist.cdf(critval)
        neither_favoured = dist.cdf(critval) - dist.cdf(-critval)
        p2_favoured = dist.cdf(-critval)

        return torch.tensor([p1_favoured, neither_favoured, p2_favoured])

    def ThetaToDefProb(self, theta):
        d = 5
        c = 3

        p = 1 / (1 + torch.exp(-1 * ((theta - d) / c)))
        return p

    def ThetaToOffProb(self, theta):
        d = 2.5
        c = 1.5

        p = 1 / (1 + torch.exp(-1 * ((theta - d) / c)))
        return p

    def SalvoBattleStochasticWrapper(self, theta: torch.tensor):
        # TODO: Distribution of remaining units have to be taken into account. Monte carlo?

        initiativeProbabilities = self.InitiativeProbabilities(theta[1, 0], theta[1, 1])

        # No ititiative
        prob_0, A1_distr, B1_distr = self.SalvoBattleStochastic(theta, theta[0, 0], theta[0, 1])

        # Initiative A
        prob_A = self.SalvoBattleStochastic(theta, theta[0, 0], B1_distr.mean)[0]

        # Initiative B
        prob_B = self.SalvoBattleStochastic(theta, A1_distr.mean, theta[0, 1])[0]

        weithed_prob = prob_0[0] * initiativeProbabilities[0] + prob_A * initiativeProbabilities[1] + prob_B * \
                       initiativeProbabilities[2]

        return weithed_prob

    def SalvoBattleStochastic(self, theta: torch.tensor, A0, B0):
        # stochastistc salvo

        theta = torch.transpose(theta, 0, -1)

        InitiativeProbs = self.InitiativeProbabilities(
            theta[1, 0], theta[1, 1])

        # Unpacking parameters
        # A0 = theta[0,0]
        # B0 = theta[0,1]

        # Player A
        # A0 = theta[0, 0]
        A_offNum = theta[2, 0]
        A_offProb = self.ThetaToOffProb(theta[3, 0])
        A_offPower = A_offNum * A_offProb
        A_defProb = self.ThetaToOffProb(theta[3, 0])
        A_defPower = theta[4, 0] * self.ThetaToDefProb(theta[5, 0])
        A_attack = theta[6, 0]
        A_stay = theta[7, 0]

        mean_gross_A = A0 * A_offPower

        A_off_distr = torch.distributions.Normal(loc=mean_gross_A, scale=A0 * A_offPower * (1 - A_offProb))
        A_def_distr = torch.distributions.Normal(loc=A0 * A_defPower, scale=A0 * A_defPower * (1 - A_defProb))

        # Player B
        # B0 = theta[0, 1]
        B_offNum = theta[2, 1]
        B_offProb = self.ThetaToOffProb(theta[3, 1])
        B_offPower = B_offNum * B_offProb
        B_defNum = theta[4, 1]
        B_defProb = self.ThetaToDefProb(theta[5, 1])
        B_defPower = B_defNum * B_defProb
        B_attack = theta[6, 1]
        B_stay = theta[7, 1]

        mu_damage_AB = A_attack / B_stay
        sigma_damage_AB = .5

        mu_damage_BA = B_attack / A_stay
        sigma_damage_BA = .5

        # Distribution of number of missile hits

        mean_net_AB = A0 * A_offPower - B0 * B_defPower
        var_net_AB = A0 * A_offPower * (1 - A_offProb) + B0 * B_defPower * (1 - B_defProb)
        AB_net_distr = torch.distributions.Normal(
            loc=mean_net_AB,
            scale=torch.sqrt(var_net_AB)
        )

        mean_net_BA = B0 * B_offPower - A0 * A_defPower
        var_net_BA = B0 * B_offPower * (1 - B_offProb) + A0 * A_defPower * (1 - A_defProb)
        BA_net_distr = torch.distributions.Normal(
            loc=mean_net_BA,
            scale=torch.sqrt(var_net_BA)
        )

        # Nominal distribution of surving units
        mean_nominal_B = B0 - mean_net_AB * mu_damage_AB
        var_nominal_B = mean_net_AB * sigma_damage_AB + var_net_AB * mu_damage_AB ** 2 - \
                        2 * sigma_damage_AB ** 2 * mean_net_AB * AB_net_distr.cdf(torch.tensor([0.0])) + \
                        2 * sigma_damage_AB ** 2 * var_net_AB * torch.exp(AB_net_distr.log_prob(torch.tensor([0.0])))

        B1_nominal_distr = torch.distributions.Normal(
            loc=mean_nominal_B,
            scale=torch.sqrt(var_nominal_B)
        )

        mean_nominal_A = A0 - mean_net_BA * mu_damage_BA
        var_nominal_A = mean_net_BA * sigma_damage_BA + var_net_BA * mu_damage_BA ** 2 - \
                        2 * sigma_damage_AB ** 2 * mean_net_BA * BA_net_distr.cdf(torch.tensor([0.0])) + \
                        2 * sigma_damage_BA ** 2 * var_net_BA * torch.exp(BA_net_distr.log_prob(torch.tensor([0.0])))
        A1_nominal_distr = torch.distributions.Normal(
            loc=mean_nominal_A,
            scale=torch.sqrt(var_nominal_A)
        )

        # We assume playerA only seeks to deny opponent full control of area
        # PlayerB seeks full control
        # Swedish military doctrine seeks to delay the opponent and survive until mobilisation of nato asssets.
        Prob_B_control = A1_nominal_distr.cdf(mu_damage_BA / 2) * (1 - B1_nominal_distr.cdf(mu_damage_AB / 2))

        return 1 - Prob_B_control, A1_nominal_distr, B1_nominal_distr

    def SalvoBattleDeterministic(self, theta: torch.tensor):
        # DeterministicSalvo
        # print(theta.requires_grad)
        theta = torch.transpose(theta, 0, -1)

        def getDeltaN(p1, p1_offPower, p1_attack, p2, p2_defPower, p2_stay):
            deltaP2 = (p1 * p1_offPower - p2 * p2_defPower) * (p1_attack / p2_stay)
            return deltaP2

        # numDraws = 1000
        # wins = [0,0]

        InitiativeProbs = self.InitiativeProbabilities(
            theta[1, 0], theta[1, 1])

        # Unpacking parameters

        A0 = theta[0, 0]
        A_offPower = theta[2, 0] * self.ThetaToOffProb(theta[3, 0])
        A_defPower = theta[4, 0] * self.ThetaToDefProb(theta[5, 0])
        A_attack = theta[6, 0]
        A_stay = theta[7, 0]

        B0 = theta[0, 1]
        B_offPower = theta[2, 1] * self.ThetaToOffProb(theta[3, 1])
        B_defPower = theta[4, 1] * self.ThetaToDefProb(theta[5, 1])
        B_attack = theta[6, 0]
        B_stay = theta[7, 0]

        # A attacks first
        deltaB = getDeltaN(A0, A_offPower, A_attack, B0, B_defPower, B_stay)
        deltaA = getDeltaN(B0 - deltaB, B_offPower, B_attack,
                           A0, A_defPower, A_stay)
        FER1 = (deltaB / B0) / (deltaA / A0)  # b-losses over a-losses

        # Simultanous fire
        deltaB = getDeltaN(A0, A_offPower, A_attack, B0, B_defPower, B_stay)
        deltaA = getDeltaN(B0, B_offPower, B_attack, A0, A_defPower, A_stay)
        FER2 = (deltaB / B0) / (deltaA / A0)  # b-losses over a-losses

        # B attacks first
        deltaB = getDeltaN(A0, A_offPower, A_attack, B0, B_defPower, B_stay)
        deltaA = getDeltaN(B0, B_offPower, B_attack, A0, A_defPower, A_stay)
        FER3 = (deltaB / B0) / (deltaA / A0)  # b-losses over a-losses

        p1_WinProb = FER1 * InitiativeProbs[0] + FER2 * \
                     InitiativeProbs[1] + FER3 * InitiativeProbs[2]
        # print(p1_WinProb)
        # returnVal = p1_WinProb * torch.tensor([1,1/p1_WinProb -1],requires_grad=True)
        return p1_WinProb

    # this should use the battle function
    def OptimizeAction(self, State, Action, max_len=torch.tensor([1, 1])):

        # this is really the only place where the whole pytorch thing is required. The rest can be base python or numpy
        iteration = 0
        learningRate = 1  # 1/16
        # torch.transpose(torch.tensor([ [-1]*self.N_Technologies , [1] * self.N_Technologies]),0,-1)
        gradFlipper = torch.tensor([-1.0, 1.0])

        act_new = Action.clone()

        stat_0 = State.clone()
        print(self.SalvoBattleStochasticWrapper(self.baseLine_params))
        winprob_0 = self.SalvoBattleStochasticWrapper(self.techToParams(stat_0))

        def scoringFun(act_n):

            stat_n = self.Update_State(stat_0, act_n)

            theta_n = self.techToParams(stat_n)
            # print(theta_n.requires_grad)

            win_prob = self.SalvoBattleStochasticWrapper(theta_n)

            # + lower_log_barrier_scaler*torch.log(act_len - eps) + upper_log_barrier_scaler*torch.log(max_len-act_len + eps)
            score_n = win_prob

            return score_n, win_prob

        while (iteration < 10):  # or torch.all(torch.norm(action_step):

            act_n = torch.tensor(act_new, requires_grad=True)  # .retain_grad()
            score_n, win_prob_n = scoringFun(act_n)

            score_n.backward(score_n)  # torch.ones_like(score_n))

            dA = act_n.grad

            action_step = gradFlipper * dA * learningRate
            act_new = torch.clamp(torch.add(act_n, action_step), 0, 1000)
            act_new = act_new / torch.sum(act_new, dim=0)

            print(
                f"norm(Action) = {torch.norm(act_new, p=2, dim=0)}, stepSize = {torch.norm(action_step, p=2, dim=0)}, winprob_0 = {winprob_0} winprob_n = {win_prob_n}")

            iteration += 1

            final_action = act_n.clone().detach()

            if final_action is not None:
                self.FINAL_ACTIONS.append(final_action)

        print(
            f"norm(Action) = {torch.norm(act_new, p=2, dim=0)}, stepSize = {torch.norm(action_step, p=2, dim=0)}, winprob_0 = {winprob_0} winprob_n = {win_prob_n}")

        return final_action

    # keep optimization trajectories that converged, and filter out "duplicates" s.t., tol < eps
    def FilterActions(self, Actions):
        # TODO : select only clustered actions, wich have converged
        return Actions[:self.N_actions]

    def GetActions(self, State):

        # ActionStartPoints = torch.rand(self.N_Technologies,2,self.N_actions_startpoint)
        P1_points = torch.transpose(self._randomPointsInSphere(
            rad=self.Plyers_action_length[0]), 0, -1)
        P2_points = torch.transpose(self._randomPointsInSphere(
            rad=self.Plyers_action_length[1]), 0, -1)
        ActionStartPoints = torch.stack((P1_points, P2_points), dim=1)

        NashEquilibria = []
        for i in range(self.N_actions_startpoint):
            init_action = ActionStartPoints[:, :, i]
            NE_action = self.OptimizeAction(
                State, init_action, self.Plyers_action_length)
            NashEquilibria.append(NE_action)

        return self.FilterActions(NashEquilibria)

    def Main(self):
        start = time.time()
        self.Q.append((self.InitialState, 0))

        node_id = 0
        while (len(self.Q) > 0 and time.time() - start < 10):

            st, t = self.Q.pop()  # the state which we are currently examining
            # print(t)
            act = self.GetActions(st)  # small number of nash equilibria
            for a in act:
                # self.History.append((st,a)) # adding the entering state and the exiting action to history, reward should probably also be added.
                # self.History.add_datapoint(st,a,nåt_anat)

                # the resulting states of traversing along the nash equilibrium
                st_new = self.Update_State(st, a)
                if t + 1 < self.Horizon:
                    self.Q.append((st_new, t + 1))
                    node_id += 1
        return self.History


if __name__ == "__main__":
    FullGame = TorchGame(Horizon=5, N_actions=8, I=25, D=-5)

    hist = FullGame.Main()
