import pandas as pd
from math import isnan
import numpy as np
import matplotlib.pyplot as plt

class Preprocessing():
    
    def __init__(self, Time_Limit_Threshold, Data_Path):
        self.Threshold = Time_Limit_Threshold
        self.Data = pd.read_csv(Data_Path, index_col=0)
        
    def Remove_NaN (self, Dict):
        """Removes NaN values"""
        NaN_Keys = []
        for key,value in Dict.items():
            if isnan(value):
                NaN_Keys.append(key)
        for NaN_Act in NaN_Keys:
            del Dict[NaN_Act] 
        return Dict 
     
    def Dictionarize(self,Data):
        """Organizes Data in dict form"""
        Dict_New={}
        values=[]
        dict_key =  ' '
        for k, v in Data.items():
            if dict_key == ' ':
                dict_key = k[0]
                
            if dict_key == k[0]:
                values.append(v)
            else: 
                Dict_New[dict_key]=values[:]
                if values == [] or len(values)==1:
                    del Dict_New[dict_key]
                del values[:]
                dict_key = k[0]
        return Dict_New

    def Mean_Algorithm(self, Times_List):
        """Returns middle and count value in each bin"""
        count=0
        z=0
        Mean=[]
        Bin = plt.hist(Times_List)
        
# Before Uncommenting, place  " , x, Label" in arguments, and when calling method set x=1 to plot and Label is the dictionary key
#        if x == 1:
#            print(Label)
#            plt.axhline(y=self.Threshold, color='r', linestyle='-')
#            plt.show()
        for i in range(0,10):       
            if Bin[0][i] > self.Threshold:
                Mean.append((Bin[1][i]+Bin[1][i+1])/2)
                count += 1
            elif Bin[0][i] > self.Threshold/1.5:
                z += 1
                if z == 2:   
                    if i<10:
                        Mean.append(Bin[1][i])
                        count += 1
                        z = 0
                    else:
                        Mean.append(Bin[1][i])
                        count += 1
                        return Mean, count
            else:
                z=0
        if count > 3:
            if len(Mean)% 2 == 0:
                Mean = [(a+b)/2 for a, b in zip(Mean[::2], Mean[1::2])]
                count = len(Mean)
            else:
                Last = Mean[-1]
                Mean = [(a+b)/2 for a, b in zip(Mean[::2], Mean[1::2])]
                Mean.append(Last)
                count = len(Mean)
            
        if not Mean :
            return Mean, count
        return Mean, count

    def Time_Limits(self, Dict, STD_Param):
        """Obtains time limit values"""
        MEAN_STD_COUNT = {}
        TIME_LIMIT = {}
        for k,v in Dict.items():
#            v = self.reject_outliers(v)
            Mean, count = self.Mean_Algorithm(v)
            if count == 0:
                MEAN_STD_COUNT[k] = ([np.mean(v)], np.std(v),1)
            else:
                MEAN_STD_COUNT[k]= (Mean,0, count) 

            for k,v in MEAN_STD_COUNT.items():
                    Mean, Std, count = v 
                    if count == 1 :
                        TIME_LIMIT[k] = [Mean[0]+ STD_Param*Std]
                    else:
                        TIME_LIMIT[k] = Mean
        """Distribution of Time Periods"""
        print("Distribution of Time Periods")
        plt.show()        
    
        return TIME_LIMIT    

    def Prep_Data_Old(self, STD_Param):
        """Obtaines Time Limits using EGO's and VCAM's approach"""

        Data = self.Data.copy(deep=True)
        Data['Duration s']=(Data['Duration ms']/1000)
        del  Data['Timestamp UTC ms'],Data['Time'], Data['Room'], Data['Steps'], Data['Move'], Data['Duration ms'],Data['Device']
            
        
        # Creating dictionary of times for each activity and setting time limit as T= mean + std-parameter*std
        Behviour_Model_Data = Data.iloc[:,0:]
        Behviour_Model_Data.reset_index(drop=True,inplace =True)
        
        
        Activity = Behviour_Model_Data.groupby(['Activity']).apply(lambda x: (x['Duration s'].mean()+ STD_Param*x['Duration s'].std())).to_dict()
        Activity_Category = Behviour_Model_Data.groupby(['Activity Category']).apply(lambda x: (x['Duration s'].mean()+ STD_Param*x['Duration s'].std() )).to_dict()
        Location = Behviour_Model_Data.groupby(['Place']).apply(lambda x: (x['Duration s'].mean()+ STD_Param*x['Duration s'].std() )).to_dict()
       
        #Remove Nan values
        Activity_Model = self.Remove_NaN(Activity)
        Activity_Category_Model = self.Remove_NaN(Activity_Category)
        Location_Model = self.Remove_NaN(Location)
    
        
        return Activity_Model, Activity_Category_Model, Location_Model
    
    def Prep_Data_New(self, STD_Param):
        """Obtaines Multiple Time Limits using Holitic Optimization Approach"""
        Data = self.Data.copy(deep=True)
        Data['Duration s']=(Data['Duration ms']/1000)
        del  Data['Timestamp UTC ms'],Data['Time'], Data['Room'], Data['Steps'], Data['Move'], Data['Duration ms'],Data['Device']
    
        
        
        # Creating dictionary of times for each state, with states as key, and list of times as values
        
        Activity = Data.groupby(['Activity']).apply(lambda x: (x['Duration s']))
        Activity_Category = Data.groupby(['Activity Category']).apply(lambda x: (x['Duration s']))
        Location = Data.groupby(['Place']).apply(lambda x: (x['Duration s']))
    
        
        
        Activity_Times= self.Dictionarize(Activity)
        Activity_Category_Times= self.Dictionarize(Activity_Category)
        Location_Times= self.Dictionarize(Location)
    
    #    Create Multiple Time limits, threshhold is used here
        Activity_Model_New= self.Time_Limits(Activity_Times, STD_Param)
        Activity_Category_Model_New= self.Time_Limits(Activity_Category_Times, STD_Param)
        Location_Model_New= self.Time_Limits(Location_Times, STD_Param)
        
        return Activity_Model_New, Activity_Category_Model_New, Location_Model_New, Activity_Times, Activity_Category_Times, Location_Times