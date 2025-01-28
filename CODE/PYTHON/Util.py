
import matplotlib.pyplot as plt
import numpy as np
from Analysis import Analysis
from Viterbi import Viterbi
from statistics import mean, stdev

Ana = Analysis

class Util:
    
    def __init__(self):
        pass
    
    def autolabel(self, rects, xpos, ax):
        """
        Attachs a text label above each bar in *rects*, displaying its height.
    
        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """
    
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0, 'right': 1, 'left': -1}
    
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(offset[xpos]*3, 3),  # use 3 points offset
                        textcoords="offset points",  # in both directions
                        ha=ha[xpos], va='bottom')
    
    
    def Plotter(self, Parameter, HOL_GROUPS, EGO_GROUPS):
        """Plots out results based on specified Parameter (Energy or Delay)"""
        """ ALL PLOTS"""
        ind1 = np.arange(len(HOL_GROUPS[Parameter]))  # the x locations for the groups
        width = 0.5  # the width of the bars
        
        fig1, ax1 = plt.subplots(constrained_layout=True, figsize=(7,4))
        rects1 = ax1.bar(ind1 - width/2, [ round(elem, 3) for elem in HOL_GROUPS[Parameter] ], width,
                        label='Holistic', color='blue')
        rects2 = ax1.bar(ind1 + width/2, [ round(elem, 3) for elem in EGO_GROUPS[Parameter] ], width,
                        label='EGO', color='red')
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax1.set_ylabel(Parameter, fontname="Arial", fontsize=14)
        ax1.set_xticks(ind1)
        ax1.tick_params(axis="y", labelsize=14)
        ax1.set_xlabel('States Combination', fontname="Arial", fontsize=14)
        ax1.set_xticklabels(str(i).zfill(2) for i in range(1,28))
        ax1.tick_params(axis="x", labelsize=14)
        ax1.legend(loc="center left", bbox_to_anchor=(0.8, 0.5), prop={'size': 14})      
        
        #Comment to remove value label from each bar
        #self.autolabel(self, rects1, "left", ax1)
        #self.autolabel(self, rects2, "right", ax1)
    
        
        """ AVGERAGE PLOTS"""
        ind2 = np.arange(1)  # the x locations for the groups
        width = 0.2  # the width of the bars
        
        fig3, ax3 = plt.subplots(constrained_layout=True, figsize=(4,4))
        rects4 = ax3.bar(ind2 - width/2, round(mean(HOL_GROUPS[Parameter]),3), width,yerr= stdev(HOL_GROUPS[Parameter]), error_kw=dict(lw=5, capsize=3, capthick=5),
                        label='Holistic', color='blue')
        rects5 = ax3.bar(ind2 + width/2, round(mean(EGO_GROUPS[Parameter]),3), width, yerr= stdev(EGO_GROUPS[Parameter]), error_kw=dict(lw=5, capsize=3, capthick=5),
                        label='EGO', color='red')
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax3.set_ylabel("Average {}".format(Parameter), fontname="Arial", fontsize=14)
        ax1.tick_params(axis="y", labelsize=14)
        Title2 = 'Average %s - EGO vs Holistic' %Parameter
        #ax3.set_title(Title2, fontname="Arial", fontsize=14)
        ax3.set_xlabel('All States', fontname="Arial", fontsize=14)
        ax3.set_xticks(ind2)
        ax3.set_xticklabels('-')
        ax3.legend(loc="center left", bbox_to_anchor=(0.8, 0.5), prop={'size': 14})      
        
        Util.autolabel(Util, rects4, "left", ax3)
        Util.autolabel(Util, rects5, "right", ax3)
    
        "Box Plot Average"
        fig2, ax2 = plt.subplots(constrained_layout=True)
        Title3 = 'Boxplots of %s - EGO vs Holistic' %Parameter
        ax2.set_title(Title3)
        ax2.boxplot([HOL_GROUPS[Parameter],EGO_GROUPS[Parameter]])
        ax2.set_xticklabels(['Holistic','EGO'])
        plt.show()
        
        plt.show()
    
    def Print_Pareto(self, title, Anotations, EGO, Pareto_Delay, Pareto_Energy, Anotate):
        """Plots out Energy vs Delay figure for all sensor group combinations"""
        fig, ax = plt.subplots(constrained_layout=True)
        
        ax.scatter(Pareto_Energy,Pareto_Delay, c='y')
        
        if Anotate == True:
            EGO_COMBO_NAME = "{}-{}-{}".format(EGO[0]['Name'],EGO[1]['Name'],EGO[2]['Name'])
            P_OBJ_arr= np.array([])
    
            for P in range(len(Pareto_Delay)):
                P_OBJ_arr=np.append(P_OBJ_arr,Pareto_Delay[P]+Pareto_Energy[P])
    
            P_Index=np.argmin(P_OBJ_arr)     
            Energy=Pareto_Energy[P_Index]
            Delay=Pareto_Delay[P_Index]
            
    #        EGO_INDEX = Anotations.index(EGO_COMBO_NAME)
            EGO_OBJ_VAL = EGO[-1]["Total Normalized Energy"],EGO[-1]["Total Normalized Energy"]
            HOL_OBJ_VAL = Delay + Energy
            for i, txt in enumerate(Anotations):
                if txt == EGO_COMBO_NAME:
                    ax.annotate(EGO_COMBO_NAME, (EGO[-1]["Total Normalized Energy"],EGO[-1]["Total Normalized Delay"]))
                else:
                    ax.annotate(txt, (Pareto_Energy[i],Pareto_Delay[i]))
    #        if Delay == Pareto_Delay[EGO_INDEX] and Energy == Pareto_Energy[EGO_INDEX]:
    #            ax.plot(Energy, Delay,'ro')
    #            print("Same Result for both methods")
    #        else: 
            print("Green for New Method, Blue for EGO")
            ax.plot(Energy, Delay, 'go')
            ax.plot(EGO[-1]["Total Normalized Energy"],EGO[-1]["Total Normalized Delay"],'bo')
                
        else:
            Delay, Energy, index = Ana.Get_Pareto(Ana, Pareto_Delay, Pareto_Energy)
            ax.plot(Energy,Delay,'ro')
#            ax.set_xlim(0.2, 1.1)
        
        ax.set_title(title)
        ax.set_ylabel('Normalized Delay')
        ax.set_xlabel('Normalized Energy')
        fig.tight_layout()
        
        plt.show()
        
        if Anotate == True:
            return EGO_OBJ_VAL, HOL_OBJ_VAL
     
    def VCAMS_Group(self, Sensor_Group, Max_Energy, Slack, Scale, Modified, Normalized,T_Norm, Pareto, Resolution, Lower, Upper, Surv_Prob_Type, Dist = []):
        """Obtaines the optimal sensing schedule and the associated energy and delay values based on sensor groups using VCAMS"""    
        if T_Norm == True:
            MAX_T = (int(Sensor_Group['Max Schedule']/Scale), True, Max_Energy)
#            Resolution=Resolution/(10**(len(str(int(MAX_T[0])))))
#            Lower=Lower/(10**(len(str(int(MAX_T[0])))))
#            Upper=Upper/(10**(len(str(int(MAX_T[0])))))
            Resolution=Resolution/int(MAX_T[0])
            Lower=Lower/int(MAX_T[0])
            Upper=Upper/int(MAX_T[0])
        else:
            MAX_T = (Sensor_Group['Max Schedule']/Scale, False, Max_Energy)
        
        if Normalized == True:
            Pareto_D,Pareto_E,alfa,beta=Ana.Solver(Ana, Resolution, Lower, Upper, Sensor_Group['Model'], Sensor_Group['Times'], Slack, Modified, MAX_T, Pareto, Sensor_Group['Norm Energy'], 1, Scale, Surv_Prob_Type, Dist)
            VIT = Viterbi(alfa,beta, Sensor_Group['Norm Energy'], 1)
        else:
            Pareto_D,Pareto_E,alfa,beta=Ana.Solver(Ana, Resolution, Lower, Upper, Sensor_Group['Model'], Sensor_Group['Times'], Slack, Modified, MAX_T, Pareto, float(sum(Sensor_Group['Energy'])), 1, Scale, Surv_Prob_Type, Dist)
            VIT = Viterbi(alfa,beta, float(sum(Sensor_Group['Energy'])), 1)
        
        print("Pareto Point for Alfa = ", alfa, " and Beta = ", beta)
        
        title = "Pareto Optimal E vs D plot for Sensor Group: %s" %Sensor_Group['Name']
    
        
        self.Print_Pareto(self, title, False, 0, Pareto_D, Pareto_E, False)
        MAX_Time = MAX_T[0]
        if MAX_T[1] == False:
            MAX_Time = 1
        if Modified == True:
            Schedule = []
            if Surv_Prob_Type != 'Distribution':
                for q in range(len(Sensor_Group['Model'])):
                    if q !=0:
                        Limit = Sensor_Group['Model'][q]-Sensor_Group['Model'][q-1]
                        Schedule += VIT.Forward(Limit, Slack, Scale, MAX_Time, Surv_Prob_Type)
                    elif q ==0:
                        Schedule += VIT.Forward(Sensor_Group['Model'][0], Slack, Scale, MAX_Time, Surv_Prob_Type)
            else:
                Last_Limit = max(Sensor_Group['Model'])
                Schedule = VIT.Forward(Last_Limit, Slack, Scale, MAX_Time, Surv_Prob_Type, Dist)
        else:
            Schedule = VIT.Forward(Sensor_Group['Model'], Slack, Scale, MAX_Time)
            
        En, De=Ana.Analyze(Ana,Sensor_Group['Times'], Sensor_Group['Model'], Schedule, Modified, float(sum(Sensor_Group['Energy'])))
        
        Sensor_Group['Schedule'], Sensor_Group['Schedule Energy'], Sensor_Group['Schedule Delay'] = Schedule, En, De
        
        Sensor_Group['Normalized Energy']=float(Sensor_Group['Schedule Energy']/(Max_Energy*(Sensor_Group['Max Schedule']/Scale)))
        
        Sensor_Group['Normalized Delay']=float(Sensor_Group['Schedule Delay']/(Sensor_Group['Max Schedule']/Scale))
        
    
    
    
    def Find_Common_Sensors(self, Groups_Combo, Max_E):
        """ Finds sensors common accros sensor group combinations and accounts for them in the energy consumption value, i.e. removes duplicate values"""
        Union={'Sensors':[],'Source':[],'Max Schedule':[], 'Max Energy':[]} 
        Synch=0
        Synch_Sensors={'Sensor':[], 'Energy':[], 'Norm Energy':[],'Group1':[],'Group2':[], 'Max Schedule':[], 'Max Energy':[]}
        Energy_Union=0 
        del Union['Sensors'][:]
        del Union['Source'][:]
        for Group in Groups_Combo:
            Sensors = Group['Sensor']
            Sensors_Energies = Group['Energy']
            Name = Group['Name']
            Max_Sched = Group['Max Schedule']
            Max_Ener = Group['Max Energy']
            for i in range(len(Sensors)):
                if Sensors[i] in Union['Sensors']:
                    Synch_Sensors['Sensor'].append(Sensors[i])
                    idx = Union['Sensors'].index(Sensors[i])
                    Synch_Sensors['Group1'].append(Union['Source'][idx])
                    Synch_Sensors['Group2'].append(Name)
                    Synch_Sensors['Energy'].append(Sensors_Energies[i])
                    Synch_Sensors['Max Energy'].append(max(Union['Max Energy'][idx], Max_Ener))
                    Synch_Sensors['Norm Energy'].append(Sensors_Energies[i]/max(Union['Max Energy'][idx], Max_Ener))
                    Synch_Sensors['Max Schedule'].append(max(Union['Max Schedule'][idx], Max_Sched))
                    Synch = 1
                else:
                    Union['Sensors'].append(Sensors[i])
                    Union['Source'].append(Name)
                    Union['Max Schedule'].append(Max_Sched)
                    Union['Max Energy'].append(Max_Ener)
                    Energy_Union+=Sensors_Energies[i]
        if Synch == 1:
            Groups_Combo.append({'Sensor':Synch_Sensors['Sensor'],'Energy':Synch_Sensors['Energy'],'Norm Energy':Synch_Sensors['Norm Energy'],'Group1':Synch_Sensors['Group1'],'Group2':Synch_Sensors['Group2'], 'Max Schedule':Synch_Sensors['Max Schedule'], 'Max Energy': Synch_Sensors['Max Energy'],'Normalized Energy':[],'Schedule':[],'Times':[],'Model':[],'Schedule Energy':[]})
        else:
            Groups_Combo.append({})
        
        return Energy_Union
    
    def Synch_Schedules(self, Schedule_1, Schedule_2, Filtered_Synch):
        """Synchronizes Sensing Schedules for common sensors accros groups recognizing different contexts"""
        if Filtered_Synch == False:
            if len(Schedule_1) > len(Schedule_2):
                for k in range(len(Schedule_2)):
                    if Schedule_2[k]=='Sense':
                        Schedule_1[k]='Sense'
                return Schedule_1
            else:
                for k in range(len(Schedule_1)):
                    if Schedule_1[k]=='Sense':
                        Schedule_2[k]='Sense'
                return Schedule_2
        elif Filtered_Synch == True:
            if len(Schedule_1) > len(Schedule_2):
                if Schedule_2[0]=='Sense':
                    Schedule_1[0]='Sense'
                for k in range(1,len(Schedule_2)):
                    if Schedule_2[k]=='Sense' and Schedule_1[k-1]=='NotSense':
                        Schedule_1[k]='Sense'
                return Schedule_1
            else:
                if Schedule_1[0]=='Sense':
                    Schedule_2[0]='Sense'
                for k in range(1,len(Schedule_1)):
                    if Schedule_1[k]=='Sense'and Schedule_2[k-1]=='NotSense':
                        Schedule_2[k]='Sense'
                return Schedule_2