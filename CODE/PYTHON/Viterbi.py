

from math import exp, log
from REWARDS import REWARDS
import numpy as np
import copy 

class Viterbi:
    
    def __init__(self, alfa, beta, energy, accuracy):
        """ Constructor for Vitebi Class, initializes parameters"""
        self.alfa=alfa
        self.beta=beta
        self.energy=energy
        self.accuracy = accuracy
        """Initialize the REWARD Function and pass the paramter values"""
        self.REW = REWARDS(self.alfa, self.beta, self.energy, self.accuracy)
        
        
    def Forward(self, T_max, Slack, Scale, MAX_T, Surv_Prob_Type ='Exponential', Dist = []):
        """1st Magical Function that creates a portion of the schedule(Up to the first sensing instance of the schedule)"""
        Actions= ['Sense', 'NotSense']
        prob =[]
        next_prob=[]
        Limit = int(T_max/Scale)
        observ=range(Limit)
        flag1 = 0
        if Surv_Prob_Type == 'Exponential':
            for t in range(Limit+100):
                prob.append(round(exp((2*log(Slack)/Limit)*t),4))
                next_prob.append(round(exp((2*log(Slack)/Limit)*(t+1)),4))
        elif Surv_Prob_Type == 'Linear':
            for t in range(Limit+10):
                pr = ((Slack-1)/Limit)*t+1
                nxt_pr = ((Slack-1)/Limit)*(t+1)+1
                prob.append(pr)
                next_prob.append(nxt_pr)
        elif Surv_Prob_Type == 'Discrete':
            for t in range(Limit+100):
                if t < Slack*Limit:
                    prob.append(Slack)
                    next_prob.append(Slack)
                elif t >= Limit - Slack*Limit:
                    prob.append(Slack)
                    next_prob.append(Slack)
                else:
                    prob.append(1)
                    next_prob.append(1)
        elif Surv_Prob_Type == 'Distribution':
            
            ys, xs = [list(yx) for yx in zip(*Dist)]
            for t in range(Limit+10):
                if t < Limit:
                      i =   np.searchsorted(xs, t, side='left')
                      pr = 1 - (ys[i-1]/max(ys))
                      prob.append(pr)    
                else:
                    prob.append(Slack)
                    
                if t + 1 < Limit:
                    j =   np.searchsorted(xs, t+1, side='left')
                    n_pr = 1 - (ys[j-1]/max(ys))  
                    next_prob.append(n_pr)
                else:
                    next_prob.append(Slack)
                    
        T = []
        temp=[]
        U = []
        argmax = []
        iterator = range(len(Actions))
        for action in Actions:
            T.append([action,0])
        for time_instant in observ:
            del U[:]
            for next_action in iterator:
                del argmax[:]
                valmax = 0
                for current_action in iterator:
                    Ti=T[current_action]
                    viterbi_path=Ti[0]
                               
                    reward = Ti[1]
                    reward = reward + self.REW.get_Mult_Reward(time_instant, current_action, next_action, prob, next_prob, MAX_T)
            
                        
                    if valmax==0:
                        valmax = reward
                        if  isinstance(viterbi_path,str):
                            temp = [viterbi_path,Actions[next_action]]
                            for te in temp:
                                argmax.append(te)
                                flag1 = 1
                        else:
                            argmax=viterbi_path[:]
                            argmax.append(Actions[next_action])
        
                    if reward >= valmax:
                        valmax = reward
                        if  isinstance(viterbi_path,str):
                            temp = [viterbi_path,Actions[next_action]]
                            argmax = temp[:]
                        else:
                            if flag1 == 0:
                                argmax=viterbi_path[:]
                                argmax.append(Actions[next_action])
                        flag1 = 0
        #                print(argmax)
                U.append([argmax[:],valmax])
            T = U[:]
        del argmax[:]   
        valmax = 0 
        for action in iterator:
            Ti=T[action]
            viterbi_path, reward = Ti[0], Ti[1]
            if valmax == 0:
                valmax = reward
            if reward >=valmax:
                argmax=viterbi_path[:]
                valmax = reward
        Result = self.Backward(-1, observ, Actions, prob, argmax,MAX_T)
        return Result
    
    def Backward(self, time_instant, observ, Actions, prob, argmax,MAX_T):
        """2nd Magical Function that creates a portion of the schedule(Up to the next sensing instance of the schedule) and is called recursively"""
        temp_f=[]
        flag = 0
        flag1 = 0
        T=[]
        U=[]        
        argmax_f = argmax[:]
        for time_inst in range(time_instant+1,len(observ)):
            if time_inst<=len(argmax) and argmax[time_inst]=="Sense":   
                flag =1
                flag1=0
                obs = range(len(observ)-time_inst)
                proba = prob[time_inst+1:]
                next_proba = prob[time_inst+2:]
                del T[:]
                iterator = range(len(Actions))
                for action in Actions:
                    T.append([action,0])
                for time_instant in range(len(obs)):
                    del U[:]
                    for next_action in iterator:
                        del argmax_f[:]
                        valmax = 0
                        for current_action in iterator:
                            Ti=T[current_action]
                            viterbi_path, reward = Ti[0], Ti[1]
                            """This is where the reward function is calculated and accumulated"""
                            reward = reward + self.REW.get_Mult_Reward(time_instant, current_action, next_action, proba, next_proba, MAX_T)
                            if valmax==0:
                                valmax = reward
                                if  isinstance(viterbi_path,str):
                                    temp_f = [viterbi_path,Actions[next_action]]
                                    for te in temp_f:
                                        argmax_f.append(te)
                                        flag1 = 1
                                else:
                                    argmax_f=viterbi_path[:]
                                    argmax_f.append(Actions[next_action])
                                    
                            if reward >= valmax:
                                valmax = reward
                                if  isinstance(viterbi_path,str):
                                    if flag1 ==0:
                                        temp_f = [viterbi_path,Actions[next_action]]
                                        argmax_f = temp_f[:]
                                        flag1 = 0
                                else:
                                    argmax_f=viterbi_path[:]
                                    if flag1 == 0:
                                        argmax_f.append(Actions[next_action])
                                    flag1 = 0
                        U.append([argmax_f[:],valmax])
                    T = U[:] 
                    
                del argmax_f[:]
                valmax = 0
                for action in iterator:
                    Ti=T[action]
                    viterbi_path, reward = Ti[0], Ti[1]
                    if valmax == 0:
                        valmax = reward
                    if reward >=valmax:
                        argmax_f = viterbi_path[:]
                        valmax = reward
                
            if flag == 1:
                argmax_temp = argmax_f[:]
                
                if len(argmax_f)<len(argmax):
                    listofsense = ["Sense"] *(time_inst)
                    argmax_f = argmax_f  + listofsense
                for j in range((len(observ)-time_inst)):
                    if j+time_inst < len(argmax_f):
                        argmax_f[j+time_inst]=argmax_temp[j]
                for k in range(time_inst):
                    if k <= len(argmax_f)-1:
                        argmax_f[k]=argmax[k]
                time_instant= copy.copy(time_inst)
                argmax=argmax_f[:]
                break
        
        if flag == 1:
            """ I had to do this stupid condition because there is a small bug that I failed to find, don't judge me"""
            if argmax[-1] =="NotSense" or argmax[-2] =="NotSense" or argmax[-3] =="NotSense" or argmax[-4] =="NotSense":
                argmax[-4]= "Sense"
                argmax[-3]= "Sense"
                argmax[-2]= "Sense"
                argmax[-1]= "Sense"
               
            argmax_f = self.Backward(time_instant, observ, Actions, prob, argmax, MAX_T)
        return argmax_f
    
   