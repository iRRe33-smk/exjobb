import pandas
import torch
import time
import json
import pandas as pd

def PCA(X : torch.Tensor):
    covs = torch.cov(X)
    covs.type(torch.complex64)
    
    vals, vec = torch.linalg.eig(covs)
    
    explained_variance = torch.cumsum(torch.abs(vals),-1) / torch.sum(torch.abs(vals))
    
    comps = vec @ covs
    
    return(vals,vec,explained_variance,comps)
    

class TorchGame():
    def __init__(self, Horizon = 5, N_actions = 5, N_actions_startpoint = 25, Start_action_length = [1,1], I=3, D = 1) -> None:
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        #torch.manual_seed(1337)
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
        
        trl = torch.pow(1+torch.exp(-State*(1/self.I)+self.D),-1)

        
        return trl

    def techToParams(self,State):
        
        trl = self.TechnologyReadiness(State)
        
        #capa_temp = torch.transpose(torch.transpose(self.PARAMCONVERSIONMATRIX,2,0),1,2)
        #2x10
        # theta = torch.matmul(trl,self.PARAMCONVERSIONMATRIX ).squeeze()
        theta = torch.matmul(self.PARAMCONVERSIONMATRIX, trl)

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
    # def SalvoBattle(self,theta):
    #     theta = torch.transpose(theta,0,-1)
    #     #deterministic salvo
    #     def getDeltaN(theta1,theta2,N1,N2):
    #         deltaP2 = (N1 * theta1[2] * theta1[3] - N2 * theta2[4] * theta2[5]) * theta1[6] / theta2[7]
    #         return deltaP2
        
        
    #     InitiativeProbs = self.InitiativeProbabilities(theta[1,0],theta[1,1])
    #     # print(np.sum(InitiativeProbs))
    #     # initiatives = np.random.choice(a = [-1,0,1], p = np.float32(InitiativeProbs), size = numDraws)

    #     # p(p1_win | p1 inititative) = 40%,
    #     # p(p1_win | no inititative) = 35%
    #     # p(p1_win | p2 inititative) = 25%
        
    #     for init in [-1,0,1]:
    #         A0 = theta[0,0]
    #         B0 = theta[0,1]
    #         if init == -1:

    #             deltaB = getDeltaN(theta[:,0], theta[:,1],A0,B0)
    #             theta[0,1] -= deltaB

    #             deltaA = getDeltaN(theta[:,1], theta[:,0],B0-deltaB,A0)
    #             theta[0,0] -= deltaA


    #         elif init == 0:
    #             deltaA = getDeltaN(theta[:,1], theta[:,0],A0,B0)
    #             deltaB = getDeltaN(theta[:,0], theta[:,1])

    #             # theta[0,0] -= deltaA
    #             # theta[0,1] -= deltaB

    #         elif init == 1:
    #             deltaA = getDeltaN(theta[:,1], theta[:,0])
    #             theta[0,0] -= deltaA

    #             deltaB = getDeltaN(theta[:,0], theta[:,1])
    #             theta[0,1] -= deltaB


    #         FER = (deltaB / B0) / (deltaA / A0) #b-losses over a-losses
    #         if FER >= 1:
    #             wins[0] += 1
    #         else:
    #             wins[1] += 1

    #     return [wins[i]/sum(wins) for i in (0,1)]


    def OptimizeAction(self, State,Action,max_len=torch.tensor([1,1])): #this should use the battle function

        #this is really the only place where the whole pytorch thing is required. The rest can be base python or numpy
        iteration = 0
        learningRate = 1/16
        gradFlipper = torch.transpose(torch.tensor([ [-1]*self.N_Technologies , [-1] * self.N_Technologies]),0,-1)


        act_new = Action.clone()
        
        
        stat_0 = State.clone()
        winprob_0 = self.S(self.techToParams(stat_0))
        
        def scoringFun(act_n):
           

            stat_n = self.Update_State(stat_0,act_n)
            
            theta_n = self.techToParams(stat_n)
            win_prob = self.Battle(theta_n)
            
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
