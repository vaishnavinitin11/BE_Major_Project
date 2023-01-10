from flask import Flask,render_template,request
import pandas as pd
import pickle

app=Flask(__name__)
file=open('RandomRegressionModel.pkl','rb')
rr=pickle.load(file)
file.close()

# life=pd.read_csv('CleanedLifeExpectancy.csv')
# @app.route('/')
# def index():
#     Countries = sorted(life['Country'].unique())
#     Status = sorted(life['Status'].unique())
#     return render_template('index.html', Countries= Countries, Status=Status)

@app.route("/",methods=["GET","POST"])
def hello_world():
    return render_template('index.html')

@app.route('/pred',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        mydict=request.form
        Country=int(mydict['Country'])
        Year=int(mydict['Year'])
        Status=int(mydict['Status'])
        AdultMortality=int(mydict['AdultMortality'])
        infantdeaths=int(mydict['infantdeaths'])
        Alcohol=float(mydict['Alcohol'])
        percentageexpenditure=float((mydict['percentageexpenditure']))
        HepatitisB=int(mydict['HepatitisB'])
        Measles=int(mydict['Measles'])
        BMI=float(mydict['BMI'])
        underFiveDeaths=int(mydict['underFiveDeaths'])
        polio=int(mydict['polio'])
        totalExpenditure=float(mydict['totalExpenditure'])
        diphtheria=int(mydict['diphtheria'])
        hivAids=float(mydict['hivAids'])
        gdp=int(mydict['gdp'])
        population=int(mydict['population'])
        thinness1_19yrs=float(mydict['thinness1_19yrs'])
        thinness5_9yrs=float(mydict['thinness5_9yrs'])
        incomeCompositionOfResources=float(mydict['incomeCompositionOfResources'])
        schooling=float(mydict['schooling'])

        inputfeatures=[Country,Year,Status,AdultMortality,infantdeaths,Alcohol,percentageexpenditure,HepatitisB,Measles,BMI,underFiveDeaths,polio,totalExpenditure,diphtheria,hivAids,gdp,population,thinness1_19yrs,thinness5_9yrs,incomeCompositionOfResources,schooling]
        
        prediction=rr.predict([inputfeatures])

        print(prediction)

        return render_template('show.html',inf=prediction)
    return render_template('index1.html')


if __name__=='__main__':
    app.run(debug=True)

