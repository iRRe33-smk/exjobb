import torch
from torch.autograd.functional import jacobian, hessian, hvp, vhp
from torch import multiprocessing as tmp
from torch.utils.data import DataLoader, Dataset, TensorDataset
# from functorch import jvp, vmap
# from functorch import jacrev as ft_jacobian, grad as ft_grad, jvp as ft_jvp
import numpy as np
from tqdm import tqdm
import time, math, json, os, random, string, multiprocessing as mp
from multiprocessing import set_start_method
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from classes.history import History
import pickle


plt.style.use('bmh')
# plt.rcParams["figure.facecolor"] = "D1E2FF"
# plt.rcParams["axes.facecolor"] = "F2F8FF"
csfont = {'fontname':'Georgia'}
hfont = {'fontname':'Helvetica'}
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times new roman"]



class RandomDataSet(Dataset):
    def __init__(self, nTech, nSamples, budget):
        self.nSamples = nSamples
        dataA = torch.rand(size = [nSamples , nTech], device="cuda")
        data_A_norm = budget[0] * dataA / torch.sum(dataA, 0)
        dataB = torch.rand(size=[nSamples, nTech], device="cuda")
        data_B_norm = budget[0] * dataB / torch.sum(dataB, 0)

        self.data = torch.cat((data_A_norm, data_B_norm), dim=1)
        print("made data")

    def __len__(self):
        return self.nSamples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx, :]
        return sample

class StateDataSet(Dataset):
    def __init__(self, State, nSamples):
        self.nSamples = nSamples
        self.State = State

    def __len__(self):
        return self.nSamples

    def __getitem__(self, idx):
        return self.State

class PseudoDistr():
    def __init__(self, loc=0, scale=0):
        self.loc = loc
        self.scale = scale

    def sample(self, unused=None):
        # return torch.stack([self.loc]*num[0],0)
        return torch.tensor([self.loc])

    def rsample(self, unused=None):
        return torch.tensor([self.loc])
    
    def cdf(self, val: torch.tensor):
        return torch.tensor([1.0])


class TorchGame():
    def __init__(self, Horizon=5, Max_actions_chosen=10, N_actions_startpoint=30,
                 Players_action_length=[10, 10], I=.5, D=5, omega=.1, Stochastic_state_update=True,
                 Max_optim_iter=50, Filter_actions=True, base_params="paper",
                 NumRepsBattle=8, DEVICE="cpu", MultiProcess=False, SGD=False,
                 fromSave=False) -> None:
        self.DEVICE = DEVICE

        self.Horizon = Horizon
        self.N_actions_startpoint = torch.tensor(N_actions_startpoint, device=self.DEVICE)
        self.Max_actions_chosen = Max_actions_chosen
        self.Players_action_length = torch.tensor(Players_action_length, device=self.DEVICE)
        self.Stochastic_state_update = Stochastic_state_update
        self.Max_optim_iter = Max_optim_iter
        self.NumRepsBattle = NumRepsBattle
        self.MultiProcess = MultiProcess
        self.SGD = SGD

        # Used in TRL calculations
        self.I = I
        self.D = D

        self.omega = omega

        self.filter_actions = Filter_actions

        self.FINAL_ACTIONS = []
        self.History = History()
        self.Q = []
        self.fromSave = fromSave

        df_stat = pd.read_excel("config_files/State_Conversion.xlsx", sheet_name="StartingState", header=0, index_col=0)
        print(df_stat)
        self.InitialState = torch.tensor(df_stat.astype(float).values, device=self.DEVICE)

        df_capaMat = pd.read_excel("config_files/State_Conversion.xlsx", sheet_name="ConversionMatrix", header=0,
                                   index_col=0)
        self.PARAMCONVERSIONMATRIX = torch.tensor(df_capaMat.astype(float).values, device=self.DEVICE)
        print(df_capaMat)

        if base_params == "custom":
            df_baseParams_paper = pd.read_excel("config_files/State_Conversion.xlsx", sheet_name="BaseParamsCustom",
                                                header=0, index_col=0)
            self.baseLine_params_paper = torch.tensor(df_baseParams_paper.astype(float).values, device=self.DEVICE)
        else:

            df_baseParams_paper = pd.read_excel("config_files/State_Conversion.xlsx", sheet_name="BaseParamsPaper",
                                                header=0, index_col=0)
            self.baseLine_params_paper = torch.tensor(df_baseParams_paper.astype(float).values, device=self.DEVICE)

        self.assign_baseline_theta()

        self.N_Capabilities, self.N_Technologies = self.PARAMCONVERSIONMATRIX.size()
        with open("citation_analysis/distribution_params.json") as f:
            dat = json.load(f)
            mu = [dat[k]["mu"] for k in dat.keys()]
            sigma = [dat[k]["scale"] for k in dat.keys()]
            self.xi_params_mu = torch.tensor(mu, device=self.DEVICE)
            self.xi_params_sigma = torch.tensor(sigma, device=self.DEVICE)
            self.xi_dist = torch.distributions.log_normal.LogNormal(self.xi_params_mu, self.xi_params_sigma)
            # print(mu, "\n", sigma)

        self.TechnologyNames = df_capaMat.columns.to_list()
        self.ParamNames = df_capaMat.index.to_list()

    def make_random_str(self, length=16):
        letters = string.ascii_lowercase
        str = ''.join([random.choice(letters) for _ in range(length)])
        return str

    def assign_baseline_theta(self):
        naive_theta_0 = self.omega * self.PARAMCONVERSIONMATRIX @ self.TechnologyReadiness(self.InitialState)
        self.baseLine_params = self.baseLine_params_paper - naive_theta_0
        return

    def _randomPointsInSphere(self, rad=1):
        # https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates

        nPoints = self.N_actions_startpoint
        nDim = self.N_Technologies
        params = (torch.rand(size=(nPoints, nDim + 1)) + 1) / 2
        # radius [0,1] * r
        # angles [0,pi/2]

        r = params[:, -1] * rad
        phi = params[:, :-1] * torch.pi / 2

        X = torch.ones(nPoints, nDim)
        for i in range(nDim - 1):
            X[:, i] *= torch.cos(phi[:, i])
            # print(f"c{i}")
            for j in range(i):
                X[:, i] *= torch.sin(phi[:, j])
                # print(f"s{j}")

        for j in range(nDim):
            X[:, nDim - 1] *= torch.sin(phi[:, j])
            # print(f"s{j}")

        X = (r * torch.eye(nPoints)) @ X  # stretching by radius
        log_X = torch.log(X)
        return log_X

    def _randomPointsInCube(self):
        nPoints = self.N_actions_startpoint
        nDim = self.N_Technologies
        return torch.rand(size=[nDim, nPoints])

    def flatten_var(self, act):
        return torch.concat((act[:, 0], act[:, 1]), 0).squeeze()

    def stack_var(self, z):
        return torch.stack((z[:self.N_Technologies], z[self.N_Technologies:]), dim=1).squeeze()

    def normAction(self, z):
        act_n = self.stack_var(torch.clamp(z, min=-10, max=25))
        # lim = 75
        # barrier = 15 * (torch.log(lim - act_n) - 5 * torch.log(torch.tensor([lim])))
        exp_act = torch.exp(act_n)  # + barrier
        act_norm = self.Players_action_length * exp_act / torch.sum(exp_act, dim=0)
        return act_norm

    def Update_State(self, State, Action, override_stochastic=None):
        if self.Stochastic_state_update:
        
            paramsize= self.xi_params_mu.size()

            xi1 = torch.exp(self.xi_params_mu + self.xi_params_sigma * torch.normal(0.0,1.0,size=paramsize))
            xi2 = torch.exp(self.xi_params_mu + self.xi_params_sigma * torch.normal(0.0,1.0,size=paramsize))
            
            xi = self.stack_var(torch.cat((xi1,xi2),dim=0))
            
            xi = torch.clamp(xi,0,5)
            
            # xi = self.xi_dist.rsample([2])
           
            
        else:
            xi = 1.0
        
        with torch.no_grad():
            if override_stochastic is not None:
                if override_stochastic:
                    xi = self.xi_dist.rsample([2]).T
                else:
                    xi = 1

        UpdateValue = self.normAction(Action) * xi
        assert ~torch.any(torch.isnan(UpdateValue))

        newState = torch.add(State, UpdateValue)
        newState = torch.clamp(newState,0,12)
        assert ~torch.any(torch.isnan(newState))

        return newState

    def TechnologyReadiness(self, State):

        trl = torch.pow(1 + torch.exp(-State / self.I + self.D), -1) + \
              torch.pow(1 + torch.exp(-State / self.I + 3 * self.D), -1)

        # trl = (1 / (1 + ((-State / self.I) + self.D).exp())) + (1 / (1 + ((-State / self.I) + 3 * self.D).exp()))

        # assert torch.allclose(trl_old, trl)

        return trl

    def techToParams(self, State):
        # print(State)
        trl = self.TechnologyReadiness(State)
        assert ~torch.any(torch.isnan(trl))
        # print(trl)
        
        theta = self.omega * (self.PARAMCONVERSIONMATRIX @ trl) + self.baseLine_params
        assert ~torch.any(torch.isnan(theta))
        # print(theta)
        return theta

    def Battle(self, theta):
        results = torch.sum(theta, dim=0) / torch.sum(theta)
        return results[0]

    def InitiativeProbabilities(self, phi, psi, sigma=1, c=1):

        stdev = 1.4142 * sigma
        # if phi - psi is None:
        #     pass #assertion
        dist = torch.distributions.Normal(loc=phi - psi, scale=stdev, validate_args=False)

        # TODO: verify parameter initiative probabiblity constant
        critval = torch.tensor(c * stdev)
        cdf_pos_critval = dist.cdf(critval)
        cdf_neg_critval = dist.cdf(-critval)

        p1_favoured = 1 - cdf_pos_critval
        neither_favoured = cdf_pos_critval - cdf_neg_critval
        p2_favoured = cdf_neg_critval

        return torch.stack([p1_favoured, neither_favoured, p2_favoured], dim=0)

    def ThetaToOffProb(self, theta):
        # TODO : verify Parameter offensive probability
        d = 0
        i = 5

        p = torch.pow(1 + torch.exp(-theta / i + d), -1)
        return p

    def ThetaToDefProb(self, theta):
        # TODO: verify parameter defensive probability
        d = 0
        i = 5

        p = torch.pow(1 + torch.exp(-theta / i + d), -1)
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

        weighted_prob = prob_0[0] * initiativeProbabilities[0] + prob_A * initiativeProbabilities[1] + prob_B * \
                        initiativeProbabilities[2]

        return weighted_prob

    def SalvoBattleSequential(self, theta: torch.tensor):

        def flipTheta(theta):
            return torch.stack((theta[:, 1], theta[:, 0]), dim=1)

        initiativeProbabilities = self.InitiativeProbabilities(theta[1, 0], theta[1, 1], c=1)
        results = torch.zeros(size=[3], dtype=torch.float64, device=self.DEVICE)

        # A init
        A_n_distr = PseudoDistr(theta[0, 0])
        B_n_distr = PseudoDistr(theta[0, 1])

        A_sample = A_n_distr.rsample().to(device=self.DEVICE)
        B_n_distr, p_B_lives, B_dead = self.getActualDefenders(theta, A_sample,
                                                               B_n_distr.rsample().to(device=self.DEVICE))
        A_n_distr, p_A_lives, A_dead = self.getActualDefenders(flipTheta(theta),
                                                               B_n_distr.rsample().to(device=self.DEVICE), A_sample)
        prob = p_A_lives
        results[0] = prob

        # No init
        A_n_distr = PseudoDistr(theta[0, 0])
        B_n_distr = PseudoDistr(theta[0, 1])

        B_sample = B_n_distr.rsample().to(device=self.DEVICE)
        A_sample = A_n_distr.rsample().to(device=self.DEVICE)
        A_n_distr, p_A_lives, A_dead = self.getActualDefenders(flipTheta(theta), B_sample, A_sample)
        # B_n_distr, p_B_lives, B_dead = self.getActualDefenders(theta, A_sample,  B_sample)
        prob = p_A_lives
        results[1] = prob

        # B init
        A_n_distr = PseudoDistr(theta[0, 0])
        B_n_distr = PseudoDistr(theta[0, 1])

        B_sample = B_n_distr.rsample().to(device=self.DEVICE)
        A_sample = A_n_distr.rsample().to(device=self.DEVICE)
        A_n_distr, p_A_lives, A_dead = self.getActualDefenders(flipTheta(theta), B_sample, A_sample)
        # B_n_distr, p_B_lives, B_dead = self.getActualDefenders(theta, A_n_distr.rsample(), B_sample)
        prob = p_A_lives
        results[2] = prob
        # print(results)
        weighted_results = torch.sum(results * initiativeProbabilities)
        return weighted_results


    def getActualDefenders(self, theta: torch.tensor, A0, B0):

        if (A0 < 0 or B0 < 0):
            #  B1_actual_distr, prob_B_lives, False
            return torch.distributions.Normal(0.0, 0.01), 0.0, True

        A_offNum = theta[2, 0]

        # d = 5
        # i = .5
        # A_offProb = torch.pow(1 + torch.exp(-theta / i + d), -1)
        A_offProb = self.ThetaToOffProb(theta[3, 0])
        A_offPower = A_offNum * A_offProb
        A_attack = theta[6, 0]

        B_defNum = theta[4, 1]

        # d = 5
        # i = .5
        # B_defProb = torch.pow(1 + torch.exp(-theta / i + d), -1)
        B_defProb = self.ThetaToDefProb(theta[5, 1])
        B_defPower = B_defNum * B_defProb
        B_stay = theta[7, 1]

        mu_damage_AB = A_attack / B_stay
        sigma_damage_AB = torch.tensor(1 / 3, device=self.DEVICE)

        mean_net_AB = A0 * A_offPower - B0 * B_defPower
        var_net_AB = A0 * A_offPower * (1 - A_offProb) + B0 * B_defPower * (1 - B_defProb)
        AB_net_distr = torch.distributions.Normal(
            loc=mean_net_AB,
            scale=torch.sqrt(var_net_AB),
            validate_args=False
        )

        mean_nominal_B = B0 - mean_net_AB * mu_damage_AB
        var_nominal_B = mean_net_AB * sigma_damage_AB ** 2 + var_net_AB * mu_damage_AB ** 2 - \
                        2 * sigma_damage_AB ** 2 * mean_net_AB * AB_net_distr.cdf(0.0) + \
                        2 * sigma_damage_AB ** 2 * var_net_AB * torch.exp(AB_net_distr.log_prob(0.0))

        B1_nominal_distr = torch.distributions.Normal(
            loc=mean_nominal_B,
            scale=torch.sqrt(var_nominal_B),
            validate_args=False
        )

        mean_actual_B = mean_nominal_B * (
                    B1_nominal_distr.cdf(B0 - mu_damage_AB / 2) - B1_nominal_distr.cdf(mu_damage_AB / 2)) - \
                        var_nominal_B * (torch.exp(B1_nominal_distr.log_prob(B0 - mu_damage_AB / 2)) - torch.exp(
            B1_nominal_distr.log_prob(mu_damage_AB / 2))) + \
                        B0 * (1 - B1_nominal_distr.cdf(B0 - mu_damage_AB / 2))  # + 1E-3

        var_actual_B = (mean_nominal_B ** 2 + var_nominal_B) * (
                    B1_nominal_distr.cdf(B0 - mu_damage_AB / 2) - B1_nominal_distr.cdf(mu_damage_AB / 2)) + \
                       B0 ** 2 * (1 - B1_nominal_distr.cdf(B0 - mu_damage_AB / 2)) - mean_actual_B ** 2 - \
                       var_nominal_B * ((B0 - mu_damage_AB / 2 + mean_nominal_B) * torch.exp(
            B1_nominal_distr.log_prob(B0 - mu_damage_AB / 2)) - \
                                        (mu_damage_AB / 2 + mean_nominal_B) * torch.exp(
                    B1_nominal_distr.log_prob(mu_damage_AB / 2)))  # + 1E-5

        # assert mean_actual_B > -.1
        # assert var_actual_B > 0

        with torch.no_grad():
            mean_actual_B = torch.clamp(mean_actual_B, min=0)
            assert var_actual_B > -1E-10
            var_actual_B = torch.clamp(var_actual_B, min=0.01)

        B1_actual_distr = torch.distributions.Normal(
            loc=mean_actual_B,
            scale=torch.sqrt(var_actual_B),
            validate_args=False
        )

        prob_B_lives = (1 - B1_nominal_distr.cdf(mu_damage_AB / 2))
        return B1_actual_distr, prob_B_lives, False

    def SalvoBattleStochastic(self, theta: torch.tensor, A0, B0):
        # stochastistc salvo

        A_offNum = theta[2, 0]
        A_offProb = self.ThetaToOffProb(theta[3, 0])
        A_offPower = A_offNum * A_offProb
        A_defProb = self.ThetaToOffProb(theta[3, 0])
        A_defPower = theta[4, 0] * self.ThetaToDefProb(theta[5, 0])
        A_attack = theta[6, 0]
        A_stay = theta[7, 0]

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

        Prob_A_lives = (1 - A1_nominal_distr.cdf(mu_damage_AB / 2))
        Prob_B_lives = (1 - B1_nominal_distr.cdf(mu_damage_BA / 2))

        # We assume playerA only seeks to deny opponent full control of area
        # PlayerB seeks full control
        # Swedish military doctrine seeks to delay the opponent and survive until mobilisation of nato asssets.
        Prob_B_control = (1 - Prob_A_lives) * Prob_B_lives

        # probability of other outcomes
        Prob_A_control = Prob_A_lives * (1 - Prob_B_lives)
        Prob_stalemate = Prob_A_lives * Prob_B_lives
        Prob_destruction = (1 - Prob_A_lives) * (1 - Prob_B_lives)

        return Prob_A_control + Prob_stalemate * .5, A1_nominal_distr, B1_nominal_distr

        # Full calculations for reference
        # Prob_A_control = (1 - A1_nominal_distr.cdf(mu_damage_AB/2)) * B1_nominal_distr.cdf(mu_damage_BA/2)
        # Prob_B_control = A1_nominal_distr.cdf(mu_damage_BA / 2) * (1 - B1_nominal_distr.cdf(mu_damage_AB / 2))
        # prob_stalemate = (1 - A1_nominal_distr.cdf(mu_damage_AB/2)) * (1- B1_nominal_distr.cdf(mu_damage_BA/2))
        # prob_destruction = (A1_nominal_distr.cdf(mu_damage_AB/2)) * B1_nominal_distr.cdf(mu_damage_BA/2)

        # return 1 - Prob_B_control, A1_nominal_distr, B1_nominal_distr

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

    def OptimizeActionIT(self, State, budget, num_acts, num_reps):
        DS = RandomDataSet(self.N_Technologies, num_acts, budget)
        DLA = DataLoader(DS, batch_size=1, shuffle=False, num_workers=0)
        DLS = DataLoader(StateDataSet(State, num_acts))

        acts = []
        for S, A in zip(DLS, DLA):
            acts.append(self.OptimizeAction(S, A, num_reps))

    def OptimizeActionMP(self, kwargs):
        return self.OptimizeAction(**kwargs)

    def OptimizeAction(self, State, Action, num_reps=8):
        stat_0 = State.clone()

        assert ~torch.all(Action == 0)
        assert ~torch.all(State == 0)

        # theta_0 = self.techToParams(stat_0)
        # winprob_0 = self.SalvoBattleSequential(theta_0)
        # print(theta_0)

        def scoringFun(z):

            score = 0
            for _ in range(num_reps):
                stat_n = self.Update_State(stat_0, z)
                assert ~torch.any(torch.isnan(stat_n))

                theta_n = self.techToParams(stat_n)
                assert ~torch.any(torch.isnan(theta_n))

                score += self.SalvoBattleSequential(theta_n)
            score /= num_reps
            # print(score)
            return score

        def T(X):
            return torch.transpose(X, 0, -1)

        def L(xi, omega):
            return xi * (1 - torch.exp(- torch.norm(omega, p=2)) ** 2)

        z_n = torch.cat((Action[:, 0], Action[:, 1]), dim=0).unsqueeze(dim=-1).requires_grad_(True)

        grad_flipper = torch.tensor(
            [1.0 if i < self.N_Technologies else -1.0 for i in range(self.N_Technologies * 2)],
            device=self.DEVICE).unsqueeze(dim=-1)

        hess_flipper = torch.zeros(size=(self.N_Technologies * 2, self.N_Technologies * 2), device=self.DEVICE)
        hess_flipper[0:self.N_Technologies, :] = 1.0
        hess_flipper[self.N_Technologies:, :] = -1.0

        params_jacobian = {
            "vectorize": True,
            "create_graph": False,
            "strict": False,
            "strategy": "reverse-mode"
        }

        params_hessian = {
            "vectorize": True,
            "create_graph": False,
            "strict": False,
            "outer_jacobian_strategy": "reverse-mode"

        }

        # block matrix [[+1,+1],
        #               [-1,-1]]

        # Hyperparmeters LSS

        def LR_sched(it, max=250, div=20, add=100):
            return torch.tensor(max / (math.exp((it + 1) / div)) + add)

        nu_n = 100 * torch.ones((self.N_Technologies * 2, 1), device=self.DEVICE)
        gamma2 = 1 * torch.tensor(1, device=self.DEVICE)  # step size nu
        xi_1 = 1 * torch.tensor(1, device=self.DEVICE)  # regularization, pushes solution towards NE
        convergence_hist = [False] * 25
        iteration = 0

        # sc = scoringFun(z_n)
        # compgraph = torchviz.make_dot(var = sc)
        # compgraph.render(directory="figures", format="png", view=True)

        winprobs = []
        grad_norms = []
        step_norms = []
        try:
            while (iteration < self.Max_optim_iter and not all(
                    convergence_hist)):  # or torch.all(torch.norm(action_step):

                with torch.no_grad():
                    if self.SGD:
                    
                        
                        jac_z = jacobian(scoringFun, z_n, **params_jacobian)
                        gamma1 = LR_sched(iteration)
                        g_x = jac_z[:self.N_Technologies]# + hess_nu[:self.N_Technologies]
                        g_y = - jac_z[self.N_Technologies:]# - hess_nu[self.N_Technologies:]
                        z_step = torch.cat((gamma1 * g_x, gamma1 * g_y), dim=0)
                        z_step = torch.clamp(z_step, min=-5, max=10)
                        z_n += z_step  # .unsqueeze(dim = 1)
                        
                        check2 = torch.norm(self.stack_var(z_step), p=2, dim=0)
                        # check2 = (torch.norm(self.stack_var(z_step), p=2, dim=0) <5E-1).all()
                        
                        convergence_hist.pop(0)
                        convergence_hist.append((check2<1E-1).all())
                        if all(convergence_hist):
                            break
                
                    else:
                        
                        jac_z = jacobian(scoringFun, z_n, **params_jacobian)
                        grad_norms.append(torch.norm(jac_z, p=2))

                        hess = hessian(lambda z: scoringFun(z).squeeze(), z_n.squeeze(), **params_hessian)
                        hess_nu = hess @ nu_n

                        omega = jac_z * grad_flipper
                        L_val = L(xi_1, omega)
                        fun_nu = lambda nu_n: torch.norm(hess_nu - omega, p=2) ** 2 + L_val * torch.norm(nu_n, p=2) ** 2
                        jac_nu = jacobian(fun_nu, nu_n, **params_jacobian)

                        g_x = jac_z[:self.N_Technologies] + hess_nu[:self.N_Technologies]
                        g_y = - jac_z[self.N_Technologies:] - hess_nu[self.N_Technologies:]
                        g_nu = jac_nu

                        gamma1 = LR_sched(iteration)
                        z_step = torch.cat((gamma1 * g_x, gamma1 * g_y), dim=0)
                        z_step = torch.clamp(z_step, min=-5, max=10)
                        z_n += z_step  # .unsqueeze(dim = 1)
                        step_norms.append(torch.norm(z_step, p=2))

                        nu_step = gamma2 * g_nu
                        nu_n += nu_step

                        check1 = torch.max(torch.abs(omega))
                        check2 = torch.norm(self.stack_var(z_step), p=2, dim=0)
                        # print(f"it: {iteration}, max(Omega): {check1}, norm(z_step): {check2} \n")
                        # print(f"norm(nu): {check2}")
                        convergence_hist.pop(0)
                        convergence_hist.append(
                            check1 < 1E-4 and torch.all(check2 < 5E-2) and iteration > 10
                        )

                        winprob_n = sum([scoringFun(z_n) for _ in range(num_reps)])/num_reps
                        winprobs.append(winprob_n)
                       
                        
                    iteration += 1

                    # if False and (iteration % 10 == 0 or iteration < 5):
                    # if iteration in [1,2]  or np.random.rand() < 0.1:
                    #     fig, axes = plt.subplots(ncols=2)
                    #     # plt.title(f"iteration: {iteration}, prob = {winprob_n}")
                    #     ax1, ax2 = axes
                    #     ax1.set_xticks([0,5,10,15,20,25])
                    #     # hess = hess + torch.randn(size=hess.size())/750
                    #     im1 = ax1.imshow((hess - torch.mean(hess, (0,1))) / torch.std(hess) , cmap="coolwarm", interpolation="nearest", label="hessian")
                    #     im2 = ax2.imshow((jac_z-torch.mean(jac_z, (0,1))) / torch.std(jac_z), cmap="coolwarm", interpolation="nearest", label="gradient")
                    #     ax2.set_xticklabels([])
                    #     # fig.legend()
                    #     fig.tight_layout(h_pad = 0, pad = .5)
                    #     fig.show()
                    #     # if input("[y] to save figure") == "y":
                    #     #     fig.savefig("figures/LSS_Hessian_2.pdf", bbox_inches = 'tight',pad_inches = 0.25)
                    #     #     print("saved figure")
                    #     # else:
                    #     #     print("did not save")
                        
                       
                    # name = f"Hessian:{iteration}_{self.make_random_str(16)}"
                    # fig.savefig(os.path.join(self.dirPath, name) +".pdf", format = "pdf")                        

            final_action = z_n.detach()

            # if np.random.rand() < 0.01:
            #     #converged hessian
            #     fig, axes = plt.subplots(ncols=2)
            #     plt.title(f"iteration: {iteration}, prob = {winprob_n}, after convergence")
            #     ax1, ax2 = axes
            #     im1 = ax1.imshow((hess - torch.mean(hess)) / torch.std(hess), cmap="hot", interpolation="nearest", label="hessian")
            #     im2 = ax2.imshow((jac_z-torch.mean(jac_z)) / torch.std(jac_z), cmap="hot", interpolation="nearest", label="gradient")
            #     name = f"Hessian_conv:{iteration}_{self.make_random_str(16)}"
            #     fig.savefig(os.path.join(self.dirPath, name) +".pdf", format = "pdf")      

            #training trajectory
            fig, ax1 = plt.subplots()
            # fig.title("norms")
            # ax1.set_yscale("log")
            ax1.set_ylabel("Gradient norms")
            ax1.set_xlabel("Iteration")
            
            ax2 = ax1.twinx()
            # ax2.set_yscale("log")
            ax2.set_ylabel("Action-step norms")
            
            L1, = ax1.plot(range(iteration), grad_norms, color="red", label="Gradient norms")
            # ax1.legend()

           
            # ax3 = ax1.twinx()
            L2, = ax2.plot(range(iteration), step_norms, color="blue", label="Action-step norms")
            fig.legend(handles = [L1, L2], loc="upper right")#, ["Gradient Norms", "Action-step Norms"])
            # ax3.plot(range(iteration), winprobs, color="green", label="winprob")
            # ax3.legend()
            # name = f"training_conv:{iteration}_{self.make_random_str(16)}"
            # plt.show()
            fileName = "LSS-convergence_0"
            path = os.path.join(os.getcwd(),"figures", fileName)+".pdf"
            # print(path) 
            plt.show()
            # plt.savefig(path, format = "pdf")
            pass

        except AssertionError as msg:
            print(msg)
            final_action = None

        return final_action

    # keep optimization trajectories that converged, and filter out "duplicates" s.t., tol < eps
    def FilterActions(self, Actions, q_max=8, share_of_variance=.8):

        def PCA(a_mat):
            (U, S, V) = torch.svd(a_mat)

            ind = ((torch.cumsum(S, 0) / torch.sum(S)) > share_of_variance).tolist()
            first_true = ind.index(True)

            q = min(q_max, first_true)
            return V[:, :q]

        # def k_means_elbow(dataset):
        #     sum_of_squared_distances = []
        #     K = range(1,10)
        #     for num_clusters in K:
        #         kmeans = KMeans(
        #             init= "random",
        #             n_clusters=num_clusters,
        #             n_init=10,
        #             max_iter=300,
        #             random_state=42
        #         )
        #         kmeans.fit(dataset)
        #         sum_of_squared_distances.append(kmeans.inertia_)
        #
        #     plt.plot(K,sum_of_squared_distances,'bx-')
        #     plt.xlabel('Values of K')
        #     plt.ylabel('Sum of squared distances/Inertia')
        #     plt.title('Elbow Method For Optimal k')
        #     plt.show()
        #
        #     optimal_k = 5
        #     return optimal_k

        def k_means_silhouette_analysis(dataset, plot):
            # max_clusters = max(self.Max_actions_chosen,)

            if plot:
                range_n_clusters = range(2, dataset.shape[1])
            else:
                range_n_clusters = range(2, self.Max_actions_chosen + 1)

            silhouette_avg = []
            for num_clusters in range_n_clusters:
                # initialise kmeans
                kmeans = KMeans(
                    init="random",
                    n_clusters=num_clusters,
                    n_init=50,
                    max_iter=500,
                    random_state=1337
                )
                kmeans.fit(dataset)
                cluster_labels = kmeans.labels_

                # silhouette score
                silhouette_avg.append(silhouette_score(dataset, cluster_labels))

            max_score = max(silhouette_avg[
                            :self.Max_actions_chosen - 2 + 1])  # -2 to compensate fort starting at 2, add 1 to get inclusive
            optimal_k = silhouette_avg.index(max_score) + 2
            # print(f"silhouette scores: {[0, 0] + silhouette_avg}")

            # if plot:
            #     plt.plot(range_n_clusters, silhouette_avg, 'bx-')
            #     plt.vlines(x=self.Max_actions_chosen,
            #                ymin=min(silhouette_avg) / 1.2, ymax=max(silhouette_avg) * 1.2,
            #                colors="r")
            #     plt.xlabel('Values of K')
            #     plt.ylabel('Silhouette score')
            #     plt.title('Silhouette analysis For Optimal k')
            #     plt.show()

            return optimal_k

        def k_means(U, plot_silhouette=False):
            u = U.detach().numpy()
            # Choose optimal number of clusters with silhouette analysis
            k = k_means_silhouette_analysis(u, plot=plot_silhouette)
            # k = k_means_elbow(u)

            kmeans = KMeans(
                init="random",
                n_clusters=max(k, 2),
                n_init=100,
                max_iter=500,
                random_state=1337
            )
            kmeans.fit(u)

            return kmeans.cluster_centers_, k

        filtered = list(filter(lambda a: ~torch.any(torch.isnan(a)), Actions))

        A = torch.cat((filtered), dim=1).T  # number of converged actions x 2*numTechs
        V = PCA(A)
        pc_projection = A @ V
        centers, n_acts = k_means(pc_projection, plot_silhouette=False)
        centers = torch.tensor(centers)
        actions = (centers @ V.T).T  # reverse projection of center points frim k-means
        acts = []
        for a in range(n_acts):
            norm_act = self.normAction(actions[:, a])
            flat = self.flatten_var(norm_act)
            acts.append(flat)
        print(f"Number of actions taken after filtration: {n_acts}")

        return acts

    def GetActionsMP(self, State):
        ActionStartPoints = torch.rand(size=[self.N_Technologies, 2, self.N_actions_startpoint], device=self.DEVICE)
        optim_params = [{"State": State.clone(), "Action": ActionStartPoints[:, :, i].clone(), "num_reps": self.NumRepsBattle} for i
                        in range(self.N_actions_startpoint)]
        if self.DEVICE == "cpu":

            processes = 15
            p = mp.pool.Pool(processes)
            chunksize = 2
            with p as pool:
                NashEquilibria = pool.map_async(self.OptimizeActionMP, optim_params, chunksize=chunksize).get()
               

        else:
            processes = 2
            p = tmp.Pool(processes,)
            chunksize = None

            with p as pool:
                NashEquilibria = pool.map_async(self.OptimizeActionMP, optim_params, chunksize=chunksize).get()


        if self.filter_actions:
            return self.FilterActions(NashEquilibria)
        else:
            return NashEquilibria[:self.Max_actions_chosen]

    def GetActions(self, State):
        # P1_points = self._randomPointsInCube()
        # P2_points = self._randomPointsInCube()
        # P1_points = torch.transpose(self._randomPointsInSphere(
        #     rad=self.Players_action_length[0]), 0, -1)
        # P2_points = torch.transpose(self._randomPointsInSphere(
        #     rad=self.Players_action_length[1]), 0, -1)
        # ActionStartPoints = torch.stack((P1_points, P2_points), dim=1)

        ActionStartPoints = torch.rand(size=[self.N_Technologies, 2, self.N_actions_startpoint], device=self.DEVICE)
        # ActionStartPoints = self.normAction(ActionStartPoints)
        NashEquilibria = []
        for i in tqdm(range(self.N_actions_startpoint),
                      desc=f"optimizing actions from {self.N_actions_startpoint} random  startingPoints", position=1,
                      leave=True):
            init_action = ActionStartPoints[:, :, i]
            NE_action = self.OptimizeAction(
                State, init_action)
            NashEquilibria.append(NE_action)
        if self.filter_actions:
            return self.FilterActions(NashEquilibria)
        else:
            return NashEquilibria[:self.Max_actions_chosen]

    
    def tempSaveRun(self, Q, hist):
        tempDirPath = "temporarySaveDir"
        
        if not os.path.isdir(tempDirPath):
            os.mkdir("temporarySaveDir")
        
        Qpath = os.path.join(tempDirPath, "Q.pkl")
        with open(Qpath,"wb+") as f:        
            pickle.dump(Q, f)
        
        histpath = os.path.join(tempDirPath, "hist.pkl")
        with open(histpath,"wb+") as f:        
            pickle.dump(hist, f)
        print("saved run")
    
    def tempLoadRun(self):
        tempDirPath = "temporarySaveDir"   
        
        Qpath = os.path.join(tempDirPath, "Q.pkl")
        with open(Qpath,"rb") as f:
            self.Q = pickle.load(f)
    
        
        histpath = os.path.join(tempDirPath, "hist.pkl")
        with open(histpath,"rb") as f:
            self.History = pickle.load(f)
        
        hlist = self.History.HistoryList
        node_idn = [el["Node_id"] for el in hlist]
        node_id = max(node_idn)
        return node_id
        
        
    def Run(self):
        
        maxactions = sum([self.Max_actions_chosen ** h for h in range(self.Horizon)])

        if self.fromSave:
            node_id = self.tempLoadRun()
            # node_id = self.Q[-1][-1]

        else:            
            node_id = 0
            self.Q.append((self.InitialState, 0, node_id))
            self.History.add_data(node_id, None, 0, self.InitialState.cpu().numpy(), None, 0)
            
        pbar = tqdm(desc="number of states evaluated", total=maxactions, position=0, leave=True)
        while (len(self.Q) > 0):
            st, t, parent_node_id = self.Q.pop()  # the state which we are currently examining

            if self.MultiProcess:
                act = self.GetActionsMP(st)  # small number of nash equilibria
            else:
                act = self.GetActions(st)

            diff = self.Max_actions_chosen - len(act)
            closed = diff * sum([self.Max_actions_chosen ** h for h in range(self.Horizon - t - 1)])
            pbar.update(len(act) + closed)

            print("\n")
            print("Q: ", len(self.Q))
            print("History: ", len(self.History.HistoryList))
            for a in act:
                st_new = self.Update_State(st, a)  # the resulting states of traversing along the nash equilibrium
                node_id += 1
                self.History.add_data(node_id, parent_node_id, t + 1, self.flatten_var(st_new).numpy(), a.numpy(), 0)

                if t + 1 < self.Horizon:
                    self.Q.append((st_new, t + 1, node_id))
                    
            self.tempSaveRun(self.Q, self.History)
        pbar.close()
        return self.History


if __name__ == "__main__":

    params_single_step = {
        "Horizon": 1, "Max_actions_chosen": 350, "N_actions_startpoint": 350, "I": .5, "D": 5,
        "Players_action_length": [1, 1], "Max_optim_iter": 64, "Filter_actions": False,
        "Stochastic_state_update": True, "base_params": "paper", "NumRepsBattle": 40,
        "DEVICE": "cpu", "MultiProcess": True, "SGD": False, "fromSave":False
    }

    params_test = {
        "Horizon": 5, "Max_actions_chosen": 2, "N_actions_startpoint": 10, "I": .5, "D": 5,
        "Players_action_length": [1, 1], "Max_optim_iter": 250, "Filter_actions": True,
        "Stochastic_state_update": True, "base_params": "custom", "NumRepsBattle": 40,
        "DEVICE": "cpu", "MultiProcess": False, "SGD": False, "fromSave":False
    }
    combitech = {
        "Horizon": 5, "Max_actions_chosen": 7, "N_actions_startpoint": 105, "I": 0.5, 
        "D": 5, "Players_action_length": [1, 1], "Max_optim_iter": 250, "Filter_actions": True, 
        "Stochastic_state_update": True, "base_params": "paper", "NumRepsBattle": 40,
        "DEVICE": "cpu", "MultiProcess": True, "fromSave":False
        }
    
    params = params_test
    FullGame = TorchGame(**params)
    
    # if FullGame.DEVICE == "cuda":
    #     set_start_method('spawn')

    hist = FullGame.Run()
    hist.save_to_file_2(params)
    hist.send_email(test=True)
