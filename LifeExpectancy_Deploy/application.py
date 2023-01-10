from flask import Flask,render_template,request
import pandas as pd
import pickle

app=Flask(__name__)
file=open('RandomRegressionModel.pkl','rb')
rr=pickle.load(file)
file.close()
life=pd.read_csv('/Users/vaishnaviuttarkar/Life Expectany/Life_Expectancy_Model/LifeExpectancy_Deploy/CleanedLifeExpectancy.csv')

@app.route('/', methods=['GET','POST'])
def index():
    countries = sorted(life['Country'].unique())
    # year = sorted(life['Country'].unique())
    Status = sorted(life['Status'].unique())
    # adultMortality
    # infantdeath

    countries.insert(0,'Select Country')
    return render_template('index.html', countries= countries, Status=Status)

@app.route('/predict',methods=['POST'])
def predict():
    country=(request.form.get('country'))
    year=int(request.form.get('year'))
    status=request.form.get('status')
    adultMortality=int(request.form.get('adultMortality'))
    infantdeaths=int(request.form.get('infantdeaths'))
    alcohol=(float((request.form.get('alcohol'))))
    percentageExpenditure=float((request.form.get('percentageExpenditure')))
    hepatitisB=int(request.form.get('hepatitisB'))
    Measles=int(request.form.get('measles'))
    bmi=float((request.form.get('bmi')))
    underFiveDeaths=int(request.form.get('underFiveDeaths'))
    polio=int(request.form.get('polio'))
    totalExpenditure=float((request.form.get('totalExpenditure')))
    diphtheria=int(request.form.get('diphtheria'))
    hivAids=float((request.form.get('hivAids')))
    gdp=int(request.form.get('gdp'))
    population=int(request.form.get('population'))
    thinness1_19yrs=float((request.form.get('thinness1_19yrs')))
    thinness5_9yrs=float((request.form.get('thinness5_9yrs')))
    incomeCompositionOfResources=float((request.form.get('incomeCompositionOfResources')))
    schooling=float((request.form.get('schooling')))


    inputfeatures=[country,year,status,adultMortality,infantdeaths,alcohol,percentageExpenditure,hepatitisB,Measles,bmi,underFiveDeaths,polio,totalExpenditure,diphtheria,hivAids,gdp,population,thinness1_19yrs,thinness5_9yrs,incomeCompositionOfResources,schooling]
    prediction=rr.predict([inputfeatures])

    print(prediction)
    return ''


if __name__=='__main__':
    app.run(debug=True)

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
    rr=RandomForestRegressor ()
    rr.fit(X_train,y_train)

    #Pickling 
    file=open('RandomRegressionModel.pkl','wb')
    pickle.dump(rr,file)
    file.close()
