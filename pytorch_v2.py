import torch
from torch.autograd import grad, Function,functional
# import numpy as np
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
    def __init__(self, N_Technologies =3, N_Capabilities = 6, Horizon = 5, N_actions = 5, N_actions_startpoint = 100, Start_action_length = [1,1], I=3, D = 1) -> None:
        #torch.manual_seed(1337)
        # global variables
        self.N_Technologies = N_Technologies
        self.N_Capabilities = N_Capabilities
        self.Horizon = Horizon
        self.N_actions_startpoint = N_actions_startpoint
        self.N_actions = N_actions
        self.Plyers_action_length = torch.tensor(Start_action_length)
        # Used in TRL calculations
        self.I = I
        self.D = D
        
        
        #load data from file
        # xi_parms = {
        #                     "mu" : [0] * self.N_Technologies,
        #                     "sigma" : [1]*self.N_Technologies
        #                   }
        with open("config_files/xi_params.json") as f:
            xi_parms = json.load(f)
        
        self.xi_params_mu = torch.tensor(xi_parms["mu"])
        self.xi_params_sigma = torch.tensor(xi_parms["sigma"])
        
        self.FINAL_ACTIONS = []
        # command and control, maneuver, intelligence, fires, sustainment, information, protection, and CIMIC
        self.CapabilityNames = ["Fires", "Protection", "Maneuver","Information","Intelligence","Sustainment","C2"]
        CapabilityMatrixShape = (2,N_Technologies,N_Capabilities)
        numElems = 2 * N_Technologies * N_Capabilities
        
        self.CAPABILITYMATRIX = torch.reshape(
            torch.normal(
                mean = torch.tensor([1/self.N_Capabilities]*numElems), 
                std= torch.tensor([0.05]*numElems)
                ),
            CapabilityMatrixShape
        )
        
        #creating the initalState
        st = torch.rand(N_Technologies,2)
        divisor = 0.01*(torch.sum(st,0)) # sum to 100
        self.InitialState = torch.divide(st,divisor)
        
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

    def TechToCapa(self,State):
        
        trl = self.TechnologyReadiness(State)
        
        #capa_temp = torch.transpose(torch.transpose(self.CAPABILITYMATRIX,2,0),1,2)
        
        capabilities = torch.matmul(trl,self.CAPABILITYMATRIX ).squeeze()
        
        return capabilities
    
    def Battle(self,Capabilities):
        results = torch.sum(Capabilities,dim=1) / torch.sum(Capabilities)
        return results
    
    def SalvoLikeBattle(self,Capabilities):
        # ["Fires", "Protection", "Maneuver","Information","Intelligence","Sustainment","C2"]
        # firstShotSkillDiff = torch.sum(Capabilities[1,[3,4,6]]) - torch.sum(Capabilities[1,[3,4,6]])
        # firstShotAdvantage = torch.tanh(firstShotSkillDiff/10)
        numParams = 5
        capToParamMatrix = torch.rand((len(self.CapabilityNames),numParams))
        
        battleParams = Capabilities @ capToParamMatrix
        
        
       
        
         
        return
    
    def OptimizeAction(self, State,Action,max_len=torch.tensor([1,1])): #this should use the battle function

        #this is really the only place where the whole pytorch thing is required. The rest can be base python or numpy
        eps = 10#1E-2
        lower_log_barrier_scaler = 0#2
        upper_log_barrier_scaler = 0#1/4
        iteration = 0
        
        learningRate = 2#1/16
        gradFlipper = torch.transpose(torch.tensor([ [1]*self.N_Technologies , [-1] * self.N_Technologies]),0,-1)

        act_new = Action.clone()
        
        
        stat_0 = State.clone()
        winprob_0 = self.Battle(self.TechToCapa(stat_0))
        
        def scoringFun(act_n):
           

            stat_n = self.Update_State(stat_0,act_n)
            
            capa_n = self.TechToCapa(stat_n)
            win_prob = self.Battle(capa_n) 
            
            score_n = win_prob #+ lower_log_barrier_scaler*torch.log(act_len - eps) + upper_log_barrier_scaler*torch.log(max_len-act_len + eps)
            
            return score_n , win_prob    
        while iteration < 500:

            act_n = torch.tensor(act_new,requires_grad=True)#.retain_grad()
            score_n , win_prob_n = scoringFun(act_n)
            
            score_n.backward(score_n)#torch.ones_like(score_n))

            dA = act_n.grad
            #hess = functional.hessian(scoringFun,act_n)
            
            
            action_step = gradFlipper * dA * learningRate
            act_new = torch.add(act_n , action_step)
            
                                

            
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
    
    nTechs = 21
    FullGame = TorchGame(N_Technologies=nTechs,Horizon=4,N_actions=5,I=1.5,D=6)
    hist = FullGame.Main()
    
    numHist = len(hist)
    
    actions = torch.zeros((numHist,nTechs*2))    

    # for i in range(numHist):
    #     act = hist[i][1]
    #     actions[i,:] = torch.flatten(act)
        
    # vals,vec,explained_variance = PCA(actions)
    
    # print(vals)
    # print(vec)
    # print(explained_variance)
        
    
    
    
    #print(len(hist))

