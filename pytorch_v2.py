import torch
# from torch.autograd import grad, Function,functional
import numpy as np
import time
import json
# import pandas as pd

def PCA(X : torch.Tensor):
    covs = torch.cov(X)
    covs.type(torch.complex64)
    
    vals, vec = torch.linalg.eig(covs)
    
    explained_variance = torch.cumsum(torch.abs(vals),-1) / torch.sum(torch.abs(vals))
    
    comps = vec @ covs
    
    return(vals,vec,explained_variance,comps)
    

class TorchGame():
    def __init__(self, Horizon = 5, N_actions = 5, N_actions_startpoint = 100, Start_action_length = [1,1], I=3, D = 1) -> None:
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        #torch.manual_seed(1337)
        # global variables
        #self.N_Technologies = N_Technologies
        self.N_BattleParams = 8
        self.Horizon = Horizon
        self.N_actions_startpoint = N_actions_startpoint
        self.N_actions = N_actions
        self.Plyers_action_length = torch.tensor(Start_action_length)
        # Used in TRL calculations
        self.I = I
        self.D = D
        
        
        #load data from file
        with open("citation_analysis/distribution_params.json") as f:
            d = json.load(f)
            mu = [d[el]["mu"] for el in d.keys()]
            sigma = [d[el]["sigma"] for el in d.keys()]
            
            self.xi_params_mu = torch.tensor(mu)
            self.xi_params_sigma = torch.tensor(sigma)
            self.N_Technologies = len(mu)
            
        

        
        self.FINAL_ACTIONS = []

        self.BattleParams_Names = ["A,B", "phi,psi", "n_a,n_b","p_a,p_b","n_y,n_z","p_y,p_z","u,v", "w,x"]
        ParamConversionMatrixShape = (2,self.N_Technologies,self.N_BattleParams)
        numElems = 2 * self.N_Technologies * self.N_BattleParams
        
        
        self.PARAMCONVERSIONMATRIX = torch.clamp(torch.reshape(
            torch.normal(
                mean = torch.tensor([1.0]*numElems), 
                std= torch.tensor([0.5]*numElems)
                ),
            ParamConversionMatrixShape
        ),min=0.1,max=3)
        
        initial_params = torch.tensor(
            [[4,7],
             [6,3], 
             [4,5],
             [.7,.6],
             [3,3],
             [.6,.6],
             [1/3,1/3],
             [1,1]
             ]
        )
        
        #This is pretty werid
        initStates = [
            torch.linalg.lstsq(
                torch.transpose(self.PARAMCONVERSIONMATRIX[i,:,:].squeeze(),0,1),
                initial_params[:,i].squeeze()).solution 
            for i in range(2)]
        self.InitialState = torch.stack(initStates,dim=-1)
        print(self.InitialState)
        
        self.History = []
        self.Q = []
    
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
                X[:,i] *= torch.sin(phi[:,j])
                #print(f"s{j}")
        
        for j in range(nDim):
                X[:,nDim-1] *= torch.sin(phi[:,j])
                #print(f"s{j}")
        
        X = (r * torch.eye(nPoints)) @ X # stretching by radius
        return X 
     
    def Update_State(self,State,Action):
        

        xi_1 = torch.normal(mean=self.xi_params_mu,std = self.xi_params_sigma)
        xi_2 = torch.normal(mean=self.xi_params_mu,std = self.xi_params_sigma)
        xi = torch.exp(torch.stack((xi_1,xi_2), dim=-1))
        UpdateValue = Action * xi
        
        newState = torch.add(State,UpdateValue)
        
        return newState

    def TechnologyReadiness(self,State):
        
        trl_temp = torch.pow(1+torch.exp(-State*(1/self.I)+self.D),-1)
        trl = torch.unsqueeze(torch.transpose(trl_temp,0,-1),1)
        
        return trl

    def techToParams(self,State):
        
        trl = self.TechnologyReadiness(State)
        
        #capa_temp = torch.transpose(torch.transpose(self.PARAMCONVERSIONMATRIX,2,0),1,2)
        #2x10
        theta = torch.matmul(trl,self.PARAMCONVERSIONMATRIX ).squeeze()
        
        return theta
    
    def Battle(self,theta):
        results = torch.sum(theta,dim=1) / torch.sum(theta)
        return results
    
    def InitiativeProbabilities(self, phi,psi, sigma=1,c=1.25):
        stdev = 1.4142*sigma
        dist = torch.distributions.Normal(loc=phi-psi,scale = stdev)
        
        critval = torch.tensor(c*stdev)
        
        p1_favoured = 1-dist.cdf(critval)
        neither_favoured = dist.cdf(critval) - dist.cdf(-critval)
        p2_favoured = dist.cdf(-critval)
        
        return [p1_favoured, neither_favoured, p2_favoured ]
    def SalvoBattle(self,theta):
        theta = torch.transpose(theta,0,-1)
        #deterministic salvo
        def getDeltaN(theta1,theta2):
            deltaP2 = (theta1[0] * theta1[2] * theta1[3] - theta2[0] * theta2[4] * theta2[5]) * theta1[6] / theta2[7]
            return deltaP2
        
        # numDraws = 1000
        # wins = [0,0]
        
        InitiativeProbs = self.InitiativeProbabilities(theta[1,0],theta[1,1])
        # print(np.sum(InitiativeProbs))
        # initiatives = np.random.choice(a = [-1,0,1], p = np.float32(InitiativeProbs), size = numDraws)
        
      
        # for i in range(numDraws):
            # init = initiatives[i]
                
            A0 = theta[0,0]
            B0 = theta[0,1]            
            if init == -1:

                deltaB = getDeltaN(theta[:,0], theta[:,1])
                theta[0,1] -= deltaB
                
                deltaA = getDeltaN(theta[:,1], theta[:,0])
                theta[0,0] -= deltaA   
                

            elif init == 0:        
                deltaA = getDeltaN(theta[:,1], theta[:,0])
                deltaB = getDeltaN(theta[:,0], theta[:,1])
                
                theta[0,0] -= deltaA    
                theta[0,1] -= deltaB
                
            elif init == 1:
                deltaA = getDeltaN(theta[:,1], theta[:,0])
                theta[0,0] -= deltaA 
                
                deltaB = getDeltaN(theta[:,0], theta[:,1])
                theta[0,1] -= deltaB
            

            FER = (deltaB / B0) / (deltaA / A0) #b-losses over a-losses
            if FER >= 1:
                wins[0] += 1
            else:
                wins[1] += 1
                
        return [wins[i]/sum(wins) for i in (0,1)]
        
        
        
        
    
        
        
       
        
         
    
    def OptimizeAction(self, State,Action,max_len=torch.tensor([1,1])): #this should use the battle function

        #this is really the only place where the whole pytorch thing is required. The rest can be base python or numpy
        iteration = 0
        learningRate = 1/16
        gradFlipper = torch.transpose(torch.tensor([ [-1]*self.N_Technologies , [-1] * self.N_Technologies]),0,-1)
        

        act_new = Action.clone()
        
        
        stat_0 = State.clone()
        winprob_0 = self.Battle(self.techToParams(stat_0))
        
        def scoringFun(act_n):
           

            stat_n = self.Update_State(stat_0,act_n)
            
            theta_n = self.techToParams(stat_n)
            win_prob = self.SalvoBattle(theta_n) 
            
            score_n = win_prob #+ lower_log_barrier_scaler*torch.log(act_len - eps) + upper_log_barrier_scaler*torch.log(max_len-act_len + eps)
            
            return score_n , win_prob    
        while (iteration < 1500):# or torch.all(torch.norm(action_step):

            act_n = torch.tensor(act_new,requires_grad=True)#.retain_grad()
            score_n , win_prob_n = scoringFun(act_n)
            
            score_n.backward(score_n)#torch.ones_like(score_n))

            dA = act_n.grad
            
            action_step = gradFlipper * dA * learningRate
            act_new = torch.add(act_n , action_step)
            act_new = act_new / torch.sum(act_new,dim=0)
            

            #print(f"norm(Action) = {torch.norm(act_new,p=2,dim=0)}, stepSize = {torch.norm(action_step,p=2,dim=0)}, winprob_0 = {winprob_0} winprob_n = {win_prob_n}")
            
           
            iteration +=1 

            final_action =act_n.clone().detach()
           
            if final_action is not None:
                self.FINAL_ACTIONS.append(final_action)
        
        
        print(f"norm(Action) = {torch.norm(act_new,p=2,dim=0)}, stepSize = {torch.norm(action_step,p=2,dim=0)}, winprob_0 = {winprob_0} winprob_n = {win_prob_n}")
            
        return final_action
        
    def FilterActions(self, Actions): #keep optimization trajectories that converged, and filter out "duplicates" s.t., tol < eps
        
        return Actions[:self.N_actions]

    def GetActions(self,State):
        
        # ActionStartPoints = torch.rand(self.N_Technologies,2,self.N_actions_startpoint)
        P1_points = torch.transpose(self._randomPointsInSphere(rad=self.Plyers_action_length[0]),0,-1)
        P2_points = torch.transpose(self._randomPointsInSphere(rad=self.Plyers_action_length[1]),0,-1)
        ActionStartPoints = torch.stack((P1_points,P2_points),dim=1)

        
        
        NashEquilibria = []
        for i in range(self.N_actions_startpoint):
            init_action = ActionStartPoints[:,:,i]
            NE_action = self.OptimizeAction(State,  init_action,self.Plyers_action_length)
            NashEquilibria.append(NE_action)
            
         
        return self.FilterActions(NashEquilibria)
    
    def Main(self):
        start = time.time()
        self.Q.append((self.InitialState,0))
        
        while (len(self.Q) > 0 and time.time() - start < 10):
            st,t = self.Q.pop() #the state which we are currently examining
            #print(t)
            act = self.GetActions(st) # small number of nash equilibria
            for a in act:
                self.History.append((st,a)) # adding the entering state and the exiting action to history, reward should probably also be added. 
                                          
                
                st_new = self.Update_State(st,a) #the resulting states of traversing along the nash equilibrium
                if t+1 < self.Horizon:
                    self.Q.append((st_new,t+1))
                    
        return self.History
                
             

if __name__ == "__main__":
    FullGame = TorchGame(Horizon=5,N_actions=8,I=1.5,D=6)
    
    hist = FullGame.Main()
    
    # print(FullGame.InitiativeProbabilities(4.0,4.0))
    # print(FullGame.InitiativeProbabilities(10.0,4.0))
    # print(FullGame.InitiativeProbabilities(14.0,14.0))
    # print(FullGame.InitiativeProbabilities(20.0,4.0))
