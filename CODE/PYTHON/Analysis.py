
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from Viterbi import Viterbi
from  itertools import product #, permutations
from math import sqrt
from operator import add
from mpl_toolkits.mplot3d import Axes3D
from kneed import KneeLocator
from tqdm import tqdm

class Analysis:
    
    def Print_Schedule(self, Schedule, Print, Energy=1, Time_Limits=0):
        """Plot out Sensing Schedules"""
        Binary_Schedule=[]
        sense_count=0
        for i in Schedule:
            if i == 'Sense':
                Binary_Schedule.append(Energy)
                sense_count+=1
            else:
                Binary_Schedule.append(0)
        if Print == True and isinstance(Time_Limits, float):
            plt.figure(num=None, figsize=(5.2, 1.2), dpi=80, facecolor='w', edgecolor='k')
            Graph=plt.plot(Binary_Schedule)
            plt.axvline(x=Time_Limits,color ='y')
            plt.yticks(np.arange(2), ("Don't\nSense", "Sense"))
            plt.xlabel('Time')
            plt.show(Graph)
        elif Print == True and isinstance(Time_Limits, list):
            plt.figure(num=None, figsize=(5.2, 1.2), dpi=80, facecolor='w', edgecolor='k')
            Graph=plt.plot(Binary_Schedule)
            for Time_Limit in Time_Limits:
                plt.axvline(x=Time_Limit,color ='y')
            plt.yticks(np.arange(2), ("Don't\nSense", "Sense"))
            plt.xlabel('Time')
            plt.show(Graph)
        else:
            return Binary_Schedule
    
            
    def Analyze(self, Times, Time_Limits, Schedule, Modified, En=1, Scale=10):
        """Obtains Energy and Delay values for the given sensing schedules"""
        Schedule = self.Print_Schedule(self, Schedule, False, En)
        Energy=[0]*len(Times)
        Delay=[0]*len(Times)
        if Modified == False:
            n=int(Time_Limits/Scale)
            for k in range(len(Times)):
                time = int(Times[k]/Scale)
                if time<=0:
                    time=1
                if time<n:
                    for j in range(time,n):
                        if Schedule[j] ==  En:
                            break
                    for i in range(j):
                        Energy[k]+=Schedule[i]
                    Delay[k]=j-time
                else:
                    Energy[k]=0
                    for i in range(n):
                        Energy[k]=Schedule[i]+Energy[k]
                    Energy[k]+=time-n
                    Delay[k]=0
            Mean_Energy=mean(Energy)
            Mean_Delay=mean(Delay)
              
#            print("Energy = ", Mean_Energy)
#            print("Delay = ", Mean_Delay)
            return Mean_Energy, Mean_Delay
        else:            
            Time_Limits= [int(t/Scale) for t in Time_Limits]
            for k in range(len(Times)):
                n=int(min(Time_Limits, key=lambda x:abs(x-Times[k]/Scale)))
                time = int(Times[k]/Scale)
                if time<=0:
                    time=1
                if time<n:
                    for j in range(time,n):
                        if Schedule[j] ==  En:
                            break
                    for i in range(j):
                        Energy[k]+=Schedule[i]
                    Delay[k]=j-time
                else:
                    Energy[k]=0
                    for i in range(n):
                        Energy[k]=Schedule[i]+Energy[k]
                    Energy[k]+=time-n
                    Delay[k]=0
            Mean_Energy=mean(Energy)
            Mean_Delay=mean(Delay)
            
#            print("Energy = ", Mean_Energy)
#            print("Delay = ", Mean_Delay)
            return Mean_Energy, Mean_Delay
    
    def pareto_frontier(self, Xs, Ys, maxX = False, maxY = False):
        """Fucntion 1 which obtains pareto frontier point"""
    # Sort the list in either ascending or descending order of X
        myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    # Start the Pareto frontier with the first value in the sorted list
        p_front = [myList[0]]    
    # Loop through the sorted list
        for pair in myList[1:]:
            if maxY: 
                if pair[1] >= p_front[-1][1]: # Look for higher values of Y…
                    p_front.append(pair) # … and add them to the Pareto frontier
            else:
                if pair[1] <= p_front[-1][1]: # Look for lower values of Y…
                    p_front.append(pair) # … and add them to the Pareto frontier
    # Turn resulting pairs back into a list of Xs and Ys
        p_frontX = [pair[0] for pair in p_front]
        p_frontY = [pair[1] for pair in p_front]
#        kneedle = KneeLocator(Xs, Ys, curve='convex', S=1.0, direction='decreasing')
#        Knee_X=kneedle.knee
#        if Knee_X is not None:
#            Knee_Y=Ys[Xs.index(kneedle.knee)]
#        else:
#            Knee_X = 0
#            Knee_Y = 0
        return p_frontX, p_frontY
    
    def Schedule_AB_printer(self, Modified, Surv_Prob_Type, Model, combo, E, D, Schedule, Scale):
        """Plots schedules for graphs at key alpha beta combinations"""
        if combo[1] > 0.8 - combo[0] and combo[1] < 1.3 - combo[0]:
            print('alpha = ', combo[0], 'beta = ', combo[1], 'E = ', round(E,3), 'D = ', round(D,3))
            if Modified == True:
                if Surv_Prob_Type == 'Distribution':
                    Last_Model = max(Model)/Scale
                    self.Print_Schedule(self,Schedule, True, 1, Last_Model)
                else:
                    Models = [x/Scale for x in Model]
                    self.Print_Schedule(self, Schedule, True, 1, Models)
            else:
                self.Print_Schedule(self, Schedule, True, 1, Model/Scale)  
            
     
    def Plot_3d(self, Comb, Energy, Delay, N_Energy, N_Delay, P_O_index):
        """3d plots for alpha-beta vs Energy, Delay, and Objective function value """
        alphions, betanions, boundary_alpha, boundary_beta, Boundary_Objective_function = [], [], [], [], []
        for c in Comb:
            alphions.append(c[0])
            betanions.append(c[1])
    
                
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        
        
        ax.plot_trisurf(alphions, betanions, Energy, cmap='viridis', linewidth=0.5, edgecolor='none');
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Beta')
        ax.set_zlabel('Energy')
        plt.show()
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        
        ax.plot_trisurf(alphions, betanions, Delay, cmap='viridis', linewidth=0.5, edgecolor='none');
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Beta')
        ax.set_zlabel('Delay')
        plt.show() 
    
        Objective_function= list(map(add, N_Energy, N_Delay))
#        for c in Comb:
#            if round(c[0],1) == 1 - round(c[1],1)  and round(c[0],1) != 0:
#                boundary_alpha.append(c[0])
#                boundary_beta.append(c[1])   
#                Boundary_Objective_function.append(N_Energy[Comb.index(c)]+N_Delay[Comb.index(c)])
#            if round(c[0],1) == 0.1:
#                boundary_alpha.append(0.1)
#                boundary_beta.append(c[1])   
#                Boundary_Objective_function.append(N_Energy[Comb.index(c)]+N_Delay[Comb.index(c)])
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        
#        ax.plot3D(boundary_alpha,boundary_beta, Boundary_Objective_function, color = 'black', linewidth=4)
        ax.scatter(Comb[P_O_index][0], Comb[P_O_index][1], Objective_function[P_O_index], color = 'brown', linewidth=4);
        
        ax.plot_trisurf(alphions, betanions, Objective_function, cmap='viridis', linewidth=0.8, edgecolor='none', alpha=0.95);
        
            
        ax.set_xlabel('Alpha', fontname="Arial", fontsize=12)
        ax.set_ylabel('Beta', fontname="Arial", fontsize=12)
        ax.set_zlabel('Objective Function', fontname="Arial", fontsize=12)
        
        plt.show()
        
            
    def Solver(self, Resolution, range_low, range_high, Model, Times, Slack, Modified, MAX_T, Pareto, energy, accuracy, Scale=10, Surv_Prob_Type='Exponential', Dist=[]):
        """Iterats over all possible combinations of alfa and beta between given lower and upper bound based on resolution (step size)"""
        Schedule, Energy, Delay, Comb= [], [], [], []
               
        alfas = np.arange(round(range_low+0.1,5),round(range_high,5), Resolution)
        betas = np.arange(round(range_low,5),round(range_high,5), Resolution)
        
        for Combos in product(alfas, betas):
            # Condition to stay within the boundary to reduce computations
            if Combos[0] + Combos[1] < 1.1:
                Comb.append((Combos[0],Combos[1]))
        
        MAX_Time = MAX_T[0]
        if MAX_T[1] == False:
            MAX_Time = 1
        
        for combination_index in tqdm(range(len(Comb))): 
            combo=list(Comb[combination_index])            
            VITER = Viterbi(round(combo[0],5),round(combo[1],5), energy, accuracy)
            if Modified == True:
                if Surv_Prob_Type != 'Distribution':
                    for q in range(len(Model)):
                        if q !=0:
                            Limit = Model[q]-Model[q-1]
                            Schedule += VITER.Forward(Limit, Slack, Scale, MAX_Time, Surv_Prob_Type)
                        elif q ==0:
                            Schedule += VITER.Forward(Model[0], Slack, Scale, MAX_Time, Surv_Prob_Type)
                else:
                    Last_Limit = max(Model)
                    Schedule = VITER.Forward(Last_Limit, Slack, Scale, MAX_Time, Surv_Prob_Type, Dist)
            else:
                Schedule = VITER.Forward(Model, Slack, Scale, MAX_Time, Surv_Prob_Type, Dist)
            
            E, D = self.Analyze(self,Times, Model, Schedule, Modified, energy, Scale)
            
#            self.Schedule_AB_printer(self, Modified, Surv_Prob_Type, Model, combo, E, D, Schedule, Scale)
            
            del Schedule[:]
            Energy.append(E)
            Delay.append(D)
        
        
        if Pareto == True:
            if max(Delay) !=0:
                Normalized_Delay = [x/max(Delay) for x in Delay]
            Normalized_Energy = [x/max(Energy) for x in Energy]
            
        
        else:
            if max(Delay) !=0:
                Normalized_Delay = [x/MAX_T[0] for x in Delay]
            Normalized_Energy = [x/(MAX_T[2]*MAX_T[0]) for x in Energy]
        
        
        
        Pareto_D, Pareto_E= self.pareto_frontier(self, Normalized_Delay, Normalized_Energy)
        
        Pareto_Opt_Delay, Pareto_Opt_Energy, Pareto_Opt_index = self.Get_Pareto(self, Normalized_Delay, Normalized_Energy)
        alfa = Comb[Pareto_Opt_index][0]
        beta = Comb[Pareto_Opt_index][1]
        
        
        self.Plot_3d(self, Comb, Energy, Delay, Normalized_Energy, Normalized_Delay, Pareto_Opt_index)
        
        if Pareto == False:
            if max(Pareto_D) !=0:
                Pareto_D = [x/max(Pareto_D) for x in Pareto_D]
            Pareto_E = [x/max(Pareto_E) for x in Pareto_E]
        
        return Pareto_D, Pareto_E, alfa, beta
    
    def Compare(self, Delay1, Energy1, Delay2, Energy2):
        """Compares combinations of energy and delay values"""    
        Distances1 = []
        Distances2 = []   
        for i in range(len(Delay1)):
            Distances1.append(sqrt(Delay1[i]**2 + Energy1[i]**2))
        for j in range(len(Delay2)):
            Distances2.append(sqrt(Delay2[j]**2 + Energy2[j]**2))
        if min(Distances1)<min(Distances2):
            return Delay1[Distances1.index(min(Distances1))], Energy1[Distances1.index(min(Distances1))], Delay2[Distances2.index(min(Distances2))], Energy2[Distances2.index(min(Distances2))], "Group 1"
        elif min(Distances2)<min(Distances1):
            return Delay2[Distances2.index(min(Distances2))], Energy2[Distances2.index(min(Distances2))], Delay1[Distances1.index(min(Distances1))], Energy1[Distances1.index(min(Distances1))], "Group 2"    
        else:
            return Delay1[Distances1.index(min(Distances1))], Energy1[Distances1.index(min(Distances1))], Delay2[Distances2.index(min(Distances2))], Energy2[Distances2.index(min(Distances2))], "Group 1&2 Same"
    
    def Get_Pareto(self, Delay, Energy):
        """Fucntion 1 which obtains pareto frontier point"""
        if len(Energy) < 2:
            Distances = []
            for i in range(len(Delay)):
                Distances.append(sqrt(Delay[i]**2 + Energy[i]**2))
            D = Delay[Distances.index(min(Distances))]
            En =  Energy[Distances.index(min(Distances))] 
        else:
            kneedle = KneeLocator(Energy, Delay, curve='convex', direction='decreasing')
            En = kneedle.knee
            if En is None:
                Distances = []
                for i in range(len(Delay)):
                    Distances.append(sqrt(Delay[i]**2 + Energy[i]**2))
                D = Delay[Distances.index(min(Distances))]
                En =  Energy[Distances.index(min(Distances))] 
            else:
                D = Delay[Energy.index(En)]
         
        Objective_index = np.argmin(list( map(add, Energy, Delay)))       
        
        
        return D, En, Objective_index