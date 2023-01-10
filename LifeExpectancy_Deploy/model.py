import pandas as pd
import numpy as np
import pickle


if __name__=='__main__':

    #importing Cleaned Dataset
    life=pd.read_csv('CleanedLifeExpectancy.csv')

    # life=life[['Country','Year','Status','AdultMortality','infantdeaths','Alcohol','percentageexpenditure','HepatitisB','Measles','BMI',underFiveDeaths,polio,totalExpenditure,diphtheria,hivAids,gdp,population,thinness1_19yrs,thinness5_9yrs,incomeCompositionOfResources,schooling]]

    #Label Encoding
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    life['Country']=le.fit_transform(life['Country'])
    life['Status']=le.fit_transform(life['Status'])


    # Separating dependent and Independent variables
    X=life.drop(columns='Lifeexpectancy')
    y=life['Lifeexpectancy']

    # Performing train-test split
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


    #Fitting algo
    from sklearn.ensemble import RandomForestRegressor
    rr=RandomForestRegressor()
    rr.fit(X_train,y_train)

    #Pickling 
    file=open('RandomRegressionModel.pkl','wb')
    pickle.dump(rr,file)
    file.close()
