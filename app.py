from pandas import DataFrame
import streamlit as st
import pickle
import sklearn


teams = ['Sunrisers Hyderabad',
            'Mumbai Indians',
            'Royal Challengers Bangalore',
            'Kolkata Knight Riders',
            'Kings XI Punjab',
            'Chennai Super Kings',
            'Rajasthan Royals',
            'Delhi Capitals'
        ]

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
        'Chandigarh', 'Jaipur', 'Chennai',
            'Centurion', 'East London',  'Kimberley',
        'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
        'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
        'Sharjah', 'Mohali', 'Bengaluru'
       ]

pipe = pickle.load(open('ipl.pkl', 'rb'))

st.title("IPL Win Predictor")

column_1, column_2 = st.columns(2)

with column_1:
    batting_team = st.selectbox('Batting Team', sorted(teams))

with column_2:
    bowling_team = st.selectbox('Bowling Team', sorted(teams))

cities = st.selectbox('Select City', sorted(cities))
score_target = st.number_input('Target', min_value=0)

column_3, column_4, column_5 = st.columns(3)

with column_3:
    current_score = st.number_input('Score', min_value=0)

with column_4:
   completed_overs =  st.number_input('Overs Completed', min_value=0, max_value=20)
    
with column_5:
    lost_wickets = st.number_input('Wickets Lost', min_value=0, max_value=10)
    
if st.button('Predict Winner'):
    runs_left = score_target - current_score
    balls_left = 120 - (completed_overs * 6)
    wickets_left = 10 - lost_wickets  
    current_run_rate = current_score/completed_overs
    required_run_rate = (runs_left * 6) / balls_left
    
    df = DataFrame({
        'batting_team' : [batting_team],
        'bowling_team' : [bowling_team],
        'city' : [cities],
        'runs_left' : [runs_left],
        'balls_left' : [balls_left],
        'wickets_left' : [wickets_left],
        'total_runs_x' : [score_target],
        'current_run_rate' : [current_run_rate],
        'required_run_rate' : [required_run_rate],
    })
    
    result = pipe.predict_proba(df)
    # st.text(result)
    loss = result[0][0]
    win = result[0][1]
    
    

    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")
    
    
