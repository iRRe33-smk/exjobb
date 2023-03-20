import torch
from torch.autograd.functional import jacobian, hessian, hvp, vhp
#from functorch import jvp, vmap
#from functorch import jacrev as ft_jacobian, grad as ft_grad, jvp as ft_jvp
# import numpy as np
import time, math, json
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from classes.history import History


class PseudoDistr():
    def __init__(self, loc = 0, scale = 0 ):
        self.loc = loc
        self.scale = scale

    def sample(self,num):
        return torch.stack([self.loc]*num[0],0)

class TorchGame():
    def __init__(self, Horizon=5, N_actions=10, N_actions_startpoint=30, Start_action_length=[1, 1], I=1, D=3, Stochastic_state_update = True) -> None:

        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        # torch.manual_seed(1337)
        # global variables
        #self.N_Technologies = N_Technologies
        self.Horizon = Horizon
        self.N_actions_startpoint = N_actions_startpoint
        self.N_actions = N_actions
        self.Players_action_length = torch.tensor(Start_action_length)
        self.Stochastic_state_update = Stochastic_state_update
        # Used in TRL calculations
        self.I = I
        self.D = D


        self.FINAL_ACTIONS = []
        self.History = History()
        self.Q = []


        df_stat = pd.read_excel("config_files/State_Conversion.xlsx", sheet_name="StartingState", header=0, index_col=0)
        print(df_stat)
        self.InitialState = torch.tensor(df_stat.astype(float).values)

        df_capaMat = pd.read_excel("config_files/State_Conversion.xlsx", sheet_name="ConversionMatrix", header=0, index_col=0)
        self.PARAMCONVERSIONMATRIX = torch.tensor(df_capaMat.astype(float).values)
        print(df_capaMat)

        df_baseParms = pd.read_excel("config_files/State_Conversion.xlsx", sheet_name="BaseParams", header=0, index_col=0)
        self.baseLine_params = torch.tensor(df_baseParms.astype(float).values)




        self.N_Capabilities,  self.N_Technologies = self.PARAMCONVERSIONMATRIX.size()

        with open("config_files/xi_params.json") as f:
            params = json.load(f)
            self.xi_params_mu = torch.cat((torch.tensor(params["mu"][:self.N_Technologies]), torch.tensor(params["mu"][:self.N_Technologies])), dim=0)
            self.xi_params_sigma = torch.cat((torch.tensor(params["sigma"][:self.N_Technologies]),torch.tensor(params["sigma"][:self.N_Technologies])), dim=0)



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
        log_X = torch.log(X)
        return log_X

    def stack_var(self, z):
        return torch.stack((z[:self.N_Technologies], z[self.N_Technologies:]), dim=1).squeeze()

    def normAction(self, z):
        act_n = self.stack_var(z)
        lim = 75
        barrier = 1 * (torch.log(lim - act_n) - torch.log(torch.tensor([lim])))
        exp_act = torch.exp(act_n) + barrier
        act_norm = exp_act * self.Players_action_length / torch.sum(exp_act, dim=0)
        return act_norm

    def Update_State(self, State, Action):

        with torch.no_grad():
            if self.Stochastic_state_update:
                xi= self.stack_var(torch.exp((torch.normal(mean = self.xi_params_mu, std = self.xi_params_sigma))))
            else:
                xi = 1.0
        UpdateValue = self.normAction(Action) * xi
        assert ~torch.any(torch.isnan(UpdateValue))
        # UpdateValue = self.normAction(Action)

        newState = torch.add(State, UpdateValue)
        assert ~torch.any(torch.isnan(newState))

        return newState

    def TechnologyReadiness(self,State):
        
        trl = torch.pow(1+torch.exp(-State * self.D + self.I), -1) + torch.pow(1+torch.exp(-State * self.D + self.I * 3), -1)

        
        return trl

    def techToParams(self, State):

        trl = self.TechnologyReadiness(State)
        assert ~torch.any(torch.isnan(trl))

        theta = (1 + torch.matmul(self.PARAMCONVERSIONMATRIX * .1, trl))  * self.baseLine_params
        assert ~torch.any(torch.isnan(theta))

        return theta

    def Battle(self, theta):
        results = torch.sum(theta, dim=0) / torch.sum(theta)
        return results[0]

    def InitiativeProbabilities(self, phi, psi, sigma=1, c=1.6):

        stdev = 1.4142 * sigma
        # if phi - psi is None:
        #     pass #assertion
        dist = torch.distributions.Normal(loc=phi - psi, scale=stdev, validate_args=False)

        #TODO: verify parameter initiative probabiblity constant
        critval = torch.tensor(c * stdev)
        cdf_pos_critval = dist.cdf(critval)
        cdf_neg_critval = dist.cdf(-critval)

        p1_favoured = 1 - cdf_pos_critval
        neither_favoured = cdf_pos_critval - cdf_neg_critval
        p2_favoured = cdf_neg_critval

        return torch.stack([p1_favoured, neither_favoured, p2_favoured], dim = 0)

    def ThetaToOffProb(self, theta):
        #TODO : verify Parameter offensive probability
        d = .5
        i = 2

        p = torch.pow(1 + torch.exp(-theta * d + i), -1)
        return p

    def ThetaToDefProb(self, theta):
        # TODO: verify parameter defensive probability
        d = .5
        i = 1.5

        p = torch.pow(1 + torch.exp(-theta * d + i), -1)
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

    def SalvoBattleIterative(self,theta: torch.tensor):

        def flipTheta(theta):
            return torch.stack((theta[:, 1], theta[:, 0]), dim=1)

        weighted_results = 0
        initiativeProbabilities = self.InitiativeProbabilities(theta[1, 0], theta[1, 1])

        numSamples = 1
        depth = 1

        # A init
        cumProb = 0
        for i in range(numSamples):
            A_n_distr = PseudoDistr(theta[0, 0])
            B_n_distr = PseudoDistr(theta[0, 1])
            for j in range(depth):
                A_sample = A_n_distr.sample([1])
                B_n_distr, p_B_lives = self.getNominalDefenders(theta, A_sample, B_n_distr.sample([1]))
                A_n_distr, p_A_lives = self.getNominalDefenders(flipTheta(theta), B_n_distr.sample([1]), A_sample)

            prob = 1 - (1 - p_A_lives.squeeze()) * p_B_lives.squeeze()  # Probability of B not winning. Any other results is good for player A
            cumProb += prob
        weighted_results += (cumProb / numSamples) * initiativeProbabilities[0]

        #No init
        cumProb = 0
        for i in range(numSamples):
            A_n_distr = PseudoDistr(theta[0,0])
            B_n_distr = PseudoDistr(theta[0,1])
            for j in range(depth):
                B_sample = B_n_distr.sample([1])
                A_sample = A_n_distr.sample([1])
                A_n_distr, p_A_lives = self.getNominalDefenders(flipTheta(theta),  B_sample, A_sample)
                B_n_distr, p_B_lives = self.getNominalDefenders(theta, A_sample,  B_sample)

            prob = 1 - (1 - p_A_lives.squeeze()) * p_B_lives.squeeze()  # Probability of B not winning. Any other results is good for player A
            cumProb += prob
        weighted_results += (cumProb / numSamples) * initiativeProbabilities[1]



        # B init
        cumProb = 0
        for i in range(numSamples):
            A_n_distr = PseudoDistr(theta[0,0])
            B_n_distr = PseudoDistr(theta[0,1])
            for j in range(depth):
                B_sample = B_n_distr.sample([1])
                A_n_distr, p_A_lives = self.getNominalDefenders(flipTheta(theta),  B_sample, A_n_distr.sample([1]))
                B_n_distr, p_B_lives = self.getNominalDefenders(theta, A_n_distr.sample([1]), B_sample)

            prob = 1 - (1 - p_A_lives.squeeze()) * p_B_lives.squeeze()  # Probability of B not winning. Any other results is good for player A
            cumProb += prob
        weighted_results += (cumProb / numSamples) * initiativeProbabilities[2]

        # B_n_distr = self.getNominalDefenders(theta, theta[0, 1], theta[0, 0])
        # A_n_distr = self.getNominalDefenders(flipTheta(theta), B_n_distr.sample(1), theta[0, 0])
        #
        # #B init
        # A_n_distr = self.getNominalDefenders(flipTheta(theta), theta[0, 1], theta[0, 0])
        # B_n_distr = self.getNominalDefenders(flipTheta(theta), A_n_distr.sample(1), theta[0, 0])

        return weighted_results

    def getNominalDefenders(self, theta: torch.tensor, A0, B0):

        with torch.no_grad():
            if A0 < 0: # Attacker was wiped out in previous salvo. No return fire. And battle definetely lost
                # B1_actual_distr, prob_B_lives
                return torch.distributions.Normal(loc=0, scale=0.01, validate_args=False), torch.tensor([0.0])

        A_offNum = theta[2, 0]
        A_offProb = self.ThetaToOffProb(theta[3, 0])
        A_offPower = A_offNum * A_offProb
        # A_defProb = self.ThetaToOffProb(theta[3, 0])
        # A_defPower = theta[4, 0] * self.ThetaToDefProb(theta[5, 0])
        A_attack = theta[6, 0]
        # A_stay = theta[7, 0]

        # B_offNum = theta[2, 1]
        # B_offProb = self.ThetaToOffProb(theta[3, 1])
        # B_offPower = B_offNum * B_offProb
        B_defNum = theta[4, 1]
        B_defProb = self.ThetaToDefProb(theta[5, 1])
        B_defPower = B_defNum * B_defProb
        # B_attack = theta[6, 1]
        B_stay = theta[7, 1]

        mu_damage_AB = A_attack / B_stay
        sigma_damage_AB = .5

        mean_net_AB = A0 * A_offPower - B0 * B_defPower
        var_net_AB = A0 * A_offPower * (1 - A_offProb) + B0 * B_defPower * (1 - B_defProb)
        AB_net_distr = torch.distributions.Normal(
            loc=mean_net_AB,
            scale=torch.sqrt(var_net_AB),
            validate_args=False
        )

        mean_nominal_B = B0 - mean_net_AB * mu_damage_AB
        var_nominal_B = mean_net_AB * sigma_damage_AB + var_net_AB * mu_damage_AB ** 2 - \
                        2 * sigma_damage_AB ** 2 * mean_net_AB * AB_net_distr.cdf(torch.tensor([0.0])) + \
                        2 * sigma_damage_AB ** 2 * var_net_AB * torch.exp(AB_net_distr.log_prob(torch.tensor([0.0])))

        B1_nominal_distr = torch.distributions.Normal(
            loc=mean_nominal_B,
            scale=torch.sqrt(var_nominal_B),
            validate_args=False
        )

        mean_actual_B = mean_nominal_B * (B1_nominal_distr.cdf(B0 - mu_damage_AB/2) - B1_nominal_distr.cdf(mu_damage_AB/2)) - \
                        var_nominal_B * (torch.exp(B1_nominal_distr.log_prob(B0 - mu_damage_AB/2)) - torch.exp(B1_nominal_distr.log_prob(mu_damage_AB/2))) + \
                        B0 *(1 - B1_nominal_distr.cdf(B0 - mu_damage_AB/2)) #+ 1E-3

        var_actual_B = (mean_nominal_B ** 2 + var_nominal_B) * ( B1_nominal_distr.cdf(B0 - mu_damage_AB/2) - B1_nominal_distr.cdf(mu_damage_AB/2)) + \
                        B0 ** 2 * (1 - B1_nominal_distr.cdf(B0 - mu_damage_AB/2)) - mean_actual_B ** 2 - \
                        var_nominal_B * ((B0 - mu_damage_AB/2 + mean_nominal_B) * torch.exp(B1_nominal_distr.log_prob(B0 - mu_damage_AB/2)) - \
                        (mu_damage_AB / 2 + mean_nominal_B) * torch.exp(B1_nominal_distr.log_prob(mu_damage_AB/2)))# + 1E-5

        # assert mean_actual_B > -.1
        # assert var_actual_B > 0

        with torch.no_grad():
            mean_actual_B = torch.clamp(mean_actual_B, min=0)
            var_actual_B = torch.clamp(var_actual_B, min=0.01)

        B1_actual_distr = torch.distributions.Normal(
            loc=mean_actual_B,
            scale = torch.sqrt(var_actual_B),
            validate_args=False
        )

        prob_B_lives = (1 - B1_nominal_distr.cdf(mu_damage_AB/2))
        # prob_B_lives_actual = 1 - B1_actual_distr.cdf(mu_damage_AB/2)
        return  B1_actual_distr, prob_B_lives

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

        Prob_A_lives = (1 - A1_nominal_distr.cdf(mu_damage_AB/2))
        Prob_B_lives = (1 - B1_nominal_distr.cdf(mu_damage_BA/2))

        # We assume playerA only seeks to deny opponent full control of area
        # PlayerB seeks full control
        # Swedish military doctrine seeks to delay the opponent and survive until mobilisation of nato asssets.
        Prob_B_control = (1 - Prob_A_lives) * Prob_B_lives

        #probability of other outcomes
        Prob_A_control = Prob_A_lives * (1 - Prob_B_lives)
        Prob_stalemate = Prob_A_lives * Prob_B_lives
        Prob_destruction = (1-Prob_A_lives) * (1 - Prob_B_lives)

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

    def OptimizeAction(self, State, Action):

        # this is really the only place where the whole pytorch thing is required. The rest can be base python or numpy

        stat_0 = State.clone()



        # print(self.SalvoBattleStochasticWrapper(self.baseLine_params))
        # winprob_0 = self.SalvoBattleStochasticWrapper(self.techToParams(stat_0))

        # def normAction(self,z):
        #     act_n = self.stack_var(z)
        #     lim = 75.0
        #     barrier = (torch.log(lim - act_n) - torch.log(torch.tensor([lim])))
        #     exp_act = torch.exp(act_n) + barrier
        #     act_norm = exp_act * self.Players_action_length / torch.sum(exp_act, dim=0)
        #     return act_norm

        def scoringFun(z):

            # act_norm = normAction(z)
            # assert ~torch.any(torch.isnan(act_norm))
            num_reps = 16
            score = 0
            for i in range(num_reps):
                stat_n = self.Update_State(stat_0, z)
                assert ~torch.any(torch.isnan(stat_n))

                theta_n = self.techToParams(stat_n)
                assert ~torch.any(torch.isnan(theta_n))

                score += self.SalvoBattleIterative(theta_n)
            score /= num_reps

            return score

        def T(X):
            return torch.transpose(X, 0, -1)

        def L(xi, omega):
            return xi * (1 - torch.exp(- torch.norm(omega, p=2))**2)

        def LR_sched(it, max=5000, div=50, add=0):
            return torch.tensor( max / (math.exp((it + 1)/div)) + add)

        z_n = torch.cat((Action[:,0], Action[:,1]), dim=0).unsqueeze(dim=-1).requires_grad_(True)
        nu_n = 0.05 * torch.ones((self.N_Technologies * 2, 1))

        grad_flipper = torch.tensor(
            [1.0 if i < self.N_Technologies else -1.0 for i in range(self.N_Technologies * 2)]
        ).unsqueeze(dim=-1)

        hess_flipper = torch.zeros(size=(self.N_Technologies * 2, self.N_Technologies * 2))
        hess_flipper[0:self.N_Technologies, :] = 1.0
        hess_flipper[self.N_Technologies:, :] = -1.0
        # block matrix [[+1,+1],
        #               [-1,-1]]


        convergence = False
        convergence_check = torch.tensor((1E-3, 1E-3, 1E-3, 1E-3))
        # gamma1 = 1E3  # learning rate
        gamma2 = 4E4  # step size nu
        xi_1 = 1e-4 * torch.tensor(1)  # regularization, pushes solution towards NE
        xi_2 = 1e-4 * torch.tensor(1)  # regularization, pushes solution towards NE

        iteration = 0

        convergence_hist = [False] * 5
        try:
            while (iteration < 250 and  not all(convergence_hist)):  # or torch.all(torch.norm(action_step):

                gamma1 = LR_sched(iteration)


                with torch.no_grad():
                    params_jacobian = {
                    "vectorize" : True,
                    "create_graph" : False,
                    "strict" : False,
                    "strategy" : "reverse-mode"
                    }
                    params_hessian = {
                    "vectorize" : True,
                    "create_graph" : False,
                    "strict" : False,
                    "outer_jacobian_strategy" : "reverse-mode"

                    }
                    jac_z = jacobian(scoringFun, z_n, **params_jacobian)

                    hess = hessian(lambda z: scoringFun(z).squeeze(), z_n.squeeze(), **params_hessian)
                    hess_nu = hess @ nu_n

                    #plot heatmap of normalized hessian.
                    # plt.imshow((hess - torch.mean(hess)) / torch.std(hess), cmap="hot", interpolation="nearest")
                    # plt.show()

                    #slower than calculating full hessian. Probabilty due to vectorization.
                    # TODO: vectorize jacobian caluclations and make hvp manually.
                    # _, hess_nu_hvp = hvp(scoringFun, z_n, nu_n)
                    # hess_nu = hess_nu_hvp

                    omega = jac_z * grad_flipper
                    L_val = L(xi_1, omega)
                    fun_nu = lambda nu: torch.norm(hess_nu - omega, p=2) ** 2 + L_val * torch.norm(nu_n, p=2) ** 2
                    jac_nu = jacobian(fun_nu, nu_n, **params_jacobian)



                    g_x = jac_z[:self.N_Technologies] + (hess_nu)[:self.N_Technologies]
                    g_y = - jac_z[self.N_Technologies:] - (hess_nu)[self.N_Technologies:]
                    g_nu = jac_nu

                    # grad, omega_nu_grad = jvp(func=lambda z: ft_jacobian(scoringFun)(z), primals=(z_n, ), tangents=(nu_n * grad_flipper, ))#
                    # omega = grad * grad_flipper
                    #
                    # g_x = (grad + omega_nu_grad)[:self.N_Technologies]
                    # g_y = (-grad + omega_nu_grad)[self.N_Technologies:]
                    #
                    # # hess = hessian(lambda z: scoringFun(z).squeeze(), z_n.squeeze(), vectorize=True) * hess_flipper
                    # # hess_nu = hess @ nu_n
                    # # block = torch.zeros((self.N_Technologies*2, self.N_Technologies*2))
                    # # block[:self.N_Technologies, :self.N_Technologies] = 1.0
                    # # block[self.N_Technologies:, self.N_Technologies:] = -1.0
                    # #
                    # # nu_select = (grad_flipper  + 1) / 2
                    # # n1 = (nu_n * nu_select) #.unsqueeze(dim=0)
                    # # n2 = (nu_n * (nu_select - 1)) #.unsqueeze(dim=0)
                    # #
                    # # #NOTE : this is only possible if game is zero sum.
                    # #
                    # # fun = lambda z: scoringFun(z).unsqueeze(dim=-1).T
                    # # hn1 = block @ ft_jvp(fun, (z_n, ), (n1, ))[1]
                    # # hn2 = block @ ft_jvp(fun, (z_n, ), (n2, ))[1]
                    # # hess_nu = block @ ft_jvp(fun, (z_n, ), (n1, ))[1] - \
                    # #           block @ ft_jvp(fun, (z_n, ), (n2, ))[1]
                    #
                    # # hess_nu = hvp(lambda z : scoringFun(z).unsqueeze(), z_n, nu_n)
                    # # fun_nu = lambda nu : torch.norm(hess_nu - omega, p=2) ** 2 + L(xi_1, omega) * torch.norm(nu_n, p=2) ** 2
                    # # g_nu = ft_jacobian(fun_nu)(nu_n)
                    #
                    # # fun_x = lambda z : scoringFun(z) + T(omega) @ nu_n
                    # # g_x = ft_jacobian(fun_x)(z_n).squeeze()[:self.N_Technologies]
                    # # fun_y = lambda z : -scoringFun(z) + T(omega) @ nu_n
                    # # g_y = ft_jacobian(fun_y)(z_n).squeeze()[self.N_Technologies:]
                    #
                    # #hess = hessian(lambda z: scoringFun(z).squeeze(), z_n.squeeze(), vectorize=True) * hess_flipper
                    # L_val =  L(xi_1, omega)
                    # fun_nu = lambda nu: torch.norm(hess @ nu - omega, p=2) ** 2 + L_val * torch.norm(nu_n, p=2) ** 2
                    # g_nu = ft_jacobian(fun_nu)(nu_n)

                    z_step = torch.cat((gamma1 * g_x, gamma1 * g_y), dim=0)
                    z_step = torch.clamp(z_step, min=-5, max=10)
                    z_n += z_step#.unsqueeze(dim = 1)

                    nu_step = gamma2 * g_nu
                    nu_n += nu_step




                    stat_n = self.Update_State(stat_0, z_n)
                    #assert ~torch.any(torch.isnan(stat_n))

                    theta_n = self.techToParams(stat_n)
                    #assert ~torch.any(torch.isnan(theta_n))

                    check1 = torch.max(torch.abs(omega))
                    check2 = torch.norm(self.stack_var(z_step), p=2, dim=0)
                    print(f"it: {iteration}, max(Omega): {check1}, norm(z_step): {check2} \n")
                    print(f"norm(nu): {torch.norm(self.stack_var(nu_n), p=2, dim=0)}")
                    convergence_hist.pop(0)
                    convergence_hist.append(
                            check1 < 1E-4 and torch.all(check2 < 1E-2)
                    )

                    iteration += 1

                    #TODO : LSS?

                    # with torch.no_grad():
                    #     omega = grad * grad_flipper
                    #     # jac = (grad * grad_flipper) # jacobian of the same point. only sign difference
                    #     hess = hessian(lambda z: scoringFun(z).squeeze(), z_n.squeeze()) * hess_flipper
                    #     #jac = jacobian(lambda z: scoringFun(z).grad, z_n)
                    #     assert ~torch.any(torch.isnan(hess)), z_n
                    #
                    #     hTv = T(hess) @ nu_n
                    #
                    #     z_step = gamma1 * (omega + torch.exp(-xi_2 * torch.norm(hTv, p=2)) * hTv)
                    #
                    #     nu_step = gamma2 * (T(hess) @ hess @ nu_n) + L(xi_1, omega) * nu_n - T(hess) @ omega
                    #
                    #     z_n += z_step
                    #     z_n.requires_grad_(True)
                    #     nu_n -= nu_step



            print(f"stopped searching after {iteration} iterations.")
            print("new action\n \n ")
            final_action = z_n.detach()
        except AssertionError as msg:
            print(msg)
            final_action = None
        # print(Action)
        # print(final_action)
        return final_action

    # keep optimization trajectories that converged, and filter out "duplicates" s.t., tol < eps
    def FilterActions(self, Actions):
        def PCA(Actions):
            a_mat = torch.cat((Actions), dim=1) #ska denna transponeras HJÄLP
            a_mat = torch.transpose(a_mat, 0, -1)
            m, n = a_mat.size()
            q = 3
            (U, S, V) = torch.pca_lowrank(a_mat, q=min(q,m,n))
            return (U, S, V)

        def k_means_elbow(dataset):
            sum_of_squared_distances = []
            K = range(1,10)
            for num_clusters in K:
                kmeans = KMeans(
                    init= "random",
                    n_clusters=num_clusters,
                    n_init=10,
                    max_iter=300,
                    random_state=42
                )
                kmeans.fit(dataset)
                sum_of_squared_distances.append(kmeans.inertia_)

            plt.plot(K,sum_of_squared_distances,'bx-')
            plt.xlabel('Values of K')
            plt.ylabel('Sum of squared distances/Inertia')
            plt.title('Elbow Method For Optimal k')
            plt.show()

            optimal_k = 5
            return optimal_k

        def k_means_siluette_analysis(dataset, plot):
            range_n_clusters = range(2,self.N_actions_startpoint)
            silhouette_avg = []
            for num_clusters in range_n_clusters:

                # initialise kmeans
                kmeans = KMeans(
                    init= "random",
                    n_clusters=num_clusters,
                    n_init=10,
                    max_iter=300,
                    random_state=42
                )
                kmeans.fit(dataset)
                cluster_labels = kmeans.labels_

                # silhouette score
                silhouette_avg.append(silhouette_score(dataset, cluster_labels))
            if plot:
                plt.plot(range_n_clusters,silhouette_avg,'bx-')
                plt.xlabel('Values of K')
                plt.ylabel('Silhouette score')
                plt.title('Silhouette analysis For Optimal k')
                plt.show()

            c = 7 # To control the maximum number of clusters the algorithm chooses, change this parameter
            max_score = max(silhouette_avg[0:c])
            optimal_k = silhouette_avg.index(max_score)+2

            return optimal_k

        def k_means(U):
            u = U.detach().numpy()
            # Choose optimal number of clusters with siluette analysis
            k = k_means_siluette_analysis(u, plot=False)
            #k = k_means_elbow(u)

            kmeans = KMeans(
                init= "random",
                n_clusters=k,
                n_init=10,
                max_iter=300,
                random_state=42
            )
            kmeans.fit(u)

            return kmeans.cluster_centers_, k

        U, S, V = PCA(Actions)
        centers, n_acts = k_means(U)
        centers = torch.tensor(centers)
        actions = torch.matmul(V, torch.transpose(centers, dim0=0, dim1=-1))
        acts = []
        for a in range(1,n_acts+1):
            acts.append(actions[:,a-1:a])

        print(f"Number of actions taken after filtration: {n_acts}")

        return acts

    def GetActions(self, State):

        # ActionStartPoints = torch.rand(self.N_Technologies,2,self.N_actions_startpoint)
        P1_points = torch.transpose(self._randomPointsInSphere(
            rad=self.Players_action_length[0]), 0, -1)
        P2_points = torch.transpose(self._randomPointsInSphere(
            rad=self.Players_action_length[1]), 0, -1)
        ActionStartPoints = torch.stack((P1_points, P2_points), dim=1)

        NashEquilibria = []
        for i in range(self.N_actions_startpoint):
            init_action = ActionStartPoints[:, :, i]
            NE_action = self.OptimizeAction(
                State, init_action)
            NashEquilibria.append(NE_action)

        return self.FilterActions(NashEquilibria)

    def Run(self):
        node_id = 0
        self.Q.append((self.InitialState, 0, node_id))
        self.History.add_data(node_id, None, 0, self.InitialState.numpy(), None, 0)

        while (len(self.Q) > 0):
            st, t, parent_node_id = self.Q.pop() #the state which we are currently examining
            act = self.GetActions(st) # small number of nash equilibria
            print("Q: ", len(self.Q))
            print("History: ", len(self.History.HistoryList))
            for a in act:
                st_new = self.Update_State(st,a) #the resulting states of traversing along the nash equilibrium
                node_id += 1
                self.History.add_data(node_id, parent_node_id, t+1, st_new.numpy(), a.numpy(), 0)

                if t+1 < self.Horizon:
                    self.Q.append((st_new,t+1, node_id))
        return self.History


if __name__ == "__main__":
    FullGame = TorchGame(Horizon=5, N_actions=3, N_actions_startpoint=100, I=5, D=2, Stochastic_state_update=True, Start_action_length=[2, 2])

    # print(FullGame.techToParams(FullGame.InitialState))
    hist = FullGame.Run()
    hist.save_to_file("test.pkl")
    #print(hist)
    # print(len(hist))



