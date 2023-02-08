import torch
from torch.autograd import grad, Function
import numpy as np
import time

class TorchGame():
    def __init__(self, N_Technologies =3, N_Capabilities = 6, Horizon = 5, N_actions = 5, N_actions_startpoint = 100, I=25, D = 0) -> None:
        torch.manual_seed(1337)
        # global variables
        self.N_Technologies = N_Technologies
        self.N_Capabilities = N_Capabilities
        self.Horizon = Horizon
        self.N_actions_startpoint = N_actions_startpoint
        self.N_actions = N_actions
        
        # Used in TRL calculations
        self.I = I
        self.D = D
        
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
        divisor = 0.01*torch.sum(st,0) # sum to 100
        self.InitialState = torch.divide(st,divisor)
        
        self.History = []
        self.Q = []
    
    def Update_State(self,State,Action):
        
        #UpdateValue = randomness(Action) #implement stochasticity
        UpdateValue = Action
        
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
    
    def OptimizeAction(self, State,Action): #this should use the battle function

        #this is really the only place where the whole pytorch thing is required. The rest can be base python or numpy
        eps = 1E-2
        iteration = 0
        
        learningRate = 1
        gradFlipper = torch.transpose(torch.tensor([ [1]*self.N_Technologies , [-1] * self.N_Technologies]),0,-1)

        act_new = Action.clone()
        
        
        stat_0 = State.clone()
        
        while iteration < 50:
            act_n = torch.tensor(act_new,requires_grad=True)
            stat_n = self.Update_State(stat_0,act_n)
            
            capa_n = self.TechToCapa(stat_n)
            score_n = self.Battle(capa_n)
            
            #######################################
            # trl_temp = torch.pow(1+torch.exp(-torch.add(State,act_n)*(1/self.I)+self.D),-1)
            # trl = torch.unsqueeze(torch.transpose(trl_temp,0,-1),1)
            
            # capa_temp = torch.transpose(torch.transpose(self.CAPABILITYMATRIX,2,0),1,2)
            # capabilities = torch.matmul(trl,capa_temp ).squeeze()
            
            # score = torch.sum(capabilities,dim=1) / torch.sum(capabilities)
            ###############################################
            
            score_n.backward(torch.ones_like(score_n))

            dA = act_n.grad

            act_new = torch.add(act_n , dA * gradFlipper * learningRate)

            
            print(f"norm(dA) = {torch.norm(dA)}, P1 winprob = {score_n}")
            
            iteration +=1 

    
        
        return (act_n.clone().detach())
        
    def FilterActions(self, Actions): #keep optimization trajectories that converged, and filter out "duplicates" s.t., tol < eps
        return Actions[:self.N_actions]

    def GetActions(self,State):
        
        ActionStartPoints = torch.rand(self.N_Technologies,2,self.N_actions_startpoint)
        
        NashEquilibria = []
        for i in range(self.N_actions_startpoint):
            init_action = ActionStartPoints[:,:,i]#.clone().detach().requires_grad_(True)
            NE_action = self.OptimizeAction(State,  init_action)
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
            
    FullGame = TorchGame(N_Technologies=21,Horizon=4,N_actions=3)
    hist = FullGame.Main()
    #print(len(hist))
