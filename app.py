import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#PAGE CONFIGURATION
st.set_page_config(
    page_title="💰Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

#LOADING MODEL AND DATA
@st.cache_resource
def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        model_data = joblib.load('salary_prediction_model.pkl')
        return model_data
    except FileNotFoundError:
        st.error("❌Model file not found! Please run the Jupyter notebook first to train the model.")
        return None

@st.cache_data
def load_sample_data():
    """Load sample data for visualization"""
    try:
        data_path = r"C:\Users\chinm\OneDrive\Desktop\IBM SkillBuild Internship\salary_prediction_project\adult 3.csv"
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        st.error("❌Dataset not found! Please check the file path.")
        return None

#Loading model and data
model_data = load_model()
sample_data = load_sample_data()

#SIDEBAR - MODEL INFORMATION
st.sidebar.markdown("# 🤖Model Information")

if model_data:
    st.sidebar.success(f"✅Model Loaded Successfully!")
    st.sidebar.info(f"**Algorithm:** {model_data['model_name']}")
    st.sidebar.info(f"**Accuracy:** {model_data['accuracy']*100:.2f}%")
    st.sidebar.info(f"**Features:** {len(model_data['feature_names'])}")
else:
    st.sidebar.error("❌Model not loaded")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊About This App")
st.sidebar.markdown("""
This app predicts whether a person's annual income 
is above or below $50,000 based on demographic and 
work-related information.

**Built with:**
- Machine Learning (Scikit-learn)
- Streamlit for web interface
- Plotly for interactive charts
""")

#MAIN PAGE HEADER
st.title("💰Employee Salary Prediction App")
st.markdown("### 🎯Predict if annual income is above or below $50,000")
st.markdown("---")

#PREDICTION FUNCTION
def make_prediction(input_data):
    """Make salary prediction using the trained model"""
    if not model_data:
        return None, None
    
    #Getting model components
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
    target_encoder = model_data['target_encoder']
    model_name = model_data['model_name']
    
    #Preparing input data
    input_df = pd.DataFrame([input_data])
    
    #Encoding categorical variables
    categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 
                          'relationship', 'race', 'gender', 'native-country']
    
    for column in categorical_columns:
        if column in input_df.columns:
            try:
                input_df[column] = label_encoders[column].transform(input_df[column].astype(str))
            except ValueError:
                #Handle unknown categories
                input_df[column] = 0
    
    #Make prediction
    try:
        if model_name == 'Support Vector Machine':
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_scaled)[0]
            else:
                probabilities = [0.5, 0.5]  #Default probabilities
        else:
            prediction = model.predict(input_df)[0]
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_df)[0]
            else:
                probabilities = [0.5, 0.5]
        
        #Converting prediction back to original label
        predicted_income = target_encoder.inverse_transform([prediction])[0]
        confidence = max(probabilities) * 100
        
        return predicted_income, confidence
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

#INPUT FORM
st.markdown("## 📝Enter Person's Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤Personal Information")
    
    age = st.number_input("Age", min_value=16, max_value=100, value=30, step=1)
    
    race = st.selectbox("Race", [
        'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'
    ])
    
    gender = st.selectbox("Gender", ['Male', 'Female'])
    
    native_country = st.selectbox("Native Country", [
        'United-States', 'Canada', 'England', 'Germany', 'Italy', 'Japan', 
        'Other'  # Simplified for demo
    ])

with col2:
    st.markdown("### 💼Work Information")
    
    workclass = st.selectbox("Work Class", [
        'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
        'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'
    ])
    
    education = st.selectbox("Education Level", [
        'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
        'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters',
        '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'
    ])
    
    occupation = st.selectbox("Occupation", [
        'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
        'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
        'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
        'Transport-moving', 'Priv-house-serv', 'Protective-serv',
        'Armed-Forces'
    ])
    
    hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40, step=1)

#Additional Information
col3, col4 = st.columns(2)

with col3:
    st.markdown("### 👨‍👩‍👧‍👦Family Information")
    
    marital_status = st.selectbox("Marital Status", [
        'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
        'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'
    ])
    
    relationship = st.selectbox("Relationship", [
        'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'
    ])

with col4:
    st.markdown("### 💰Financial Information")
    
    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0, step=100)
    
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=10000, value=0, step=50)
    
    fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1500000, value=200000, step=10000,
                            help="This represents the number of people the census believes the entry represents")
    
    educational_num = st.number_input("Education Number", min_value=1, max_value=16, value=10, step=1,
                                    help="Numerical representation of education level")

#PREDICTION SECTION
st.markdown("---")
st.markdown("## 🎯Prediction Results")

if st.button("🔮Predict Salary", type="primary", use_container_width=True):
    #Preparing input data
    input_data = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'education': education,
        'educational-num': educational_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }
    
    #Making prediction
    prediction, confidence = make_prediction(input_data)
    
    if prediction is not None:
        #Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == '>50K':
                st.success(f"💰**Predicted Income: {prediction}**")
                st.balloons()
            else:
                st.info(f"💼**Predicted Income: {prediction}**")
        
        with col2:
            st.metric("Confidence Level", f"{confidence:.1f}%")
        
        with col3:
            if confidence > 80:
                st.success("High Confidence")
            elif confidence > 60:
                st.warning("Medium Confidence")
            else:
                st.error("Low Confidence")
        
        #Visualization of prediction
        st.markdown("### 📊Prediction Visualization")
        
        #Creating gauge chart for confidence
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Prediction Confidence"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        
        fig_gauge.update_layout(height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        #Feature summary
        st.markdown("### 📋Input Summary")
        summary_data = {
            'Feature': ['Age', 'Education', 'Work Class', 'Occupation', 'Hours/Week', 'Marital Status'],
            'Value': [age, education, workclass, occupation, hours_per_week, marital_status]
        }
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

#DATA INSIGHTS SECTION
if sample_data is not None:
    st.markdown("---")
    st.markdown("## 📈Dataset Insights")
    
    tab1, tab2, tab3 = st.tabs(["Income Distribution", "Age Analysis", "Education Impact"])
    
    with tab1:
        st.markdown("### Income Distribution in Dataset")
        income_counts = sample_data['income'].value_counts()
        
        fig_pie = px.pie(
            values=income_counts.values, 
            names=income_counts.index,
            title="Distribution of Income Categories",
            color_discrete_map={'<=50K': '#ff7f7f', '>50K': '#7fbf7f'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("≤$50K Count", f"{income_counts['<=50K']:,}")
        with col2:
            st.metric(">$50K Count", f"{income_counts['>50K']:,}")
    
    with tab2:
        st.markdown("### Age Distribution by Income")
        fig_box = px.box(
            sample_data, 
            x='income', 
            y='age',
            title="Age Distribution by Income Category",
            color='income'
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        #Age statistics
        age_stats = sample_data.groupby('income')['age'].agg(['mean', 'median', 'std']).round(1)
        st.dataframe(age_stats, use_container_width=True)
    
    with tab3:
        st.markdown("### Education Level Impact")
        education_income = pd.crosstab(sample_data['education'], sample_data['income'])
        
        fig_bar = px.bar(
            x=education_income.index,
            y=[education_income['<=50K'], education_income['>50K']],
            title="Education Level vs Income",
            labels={'x': 'Education Level', 'y': 'Count'},
            color_discrete_map={'<=50K': '#ff7f7f', '>50K': '#7fbf7f'}
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

#MODEL PERFORMANCE SECTION
if model_data:
    st.markdown("---")
    st.markdown("## 🎯Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Model Accuracy",
            f"{model_data['accuracy']*100:.1f}%",
            help="Percentage of correct predictions on test data"
        )
    
    with col2:
        st.metric(
            "Algorithm Used",
            model_data['model_name'],
            help="The machine learning algorithm that performed best"
        )
    
    with col3:
        st.metric(
            "Features Used",
            len(model_data['feature_names']),
            help="Number of input features used by the model"
        )

#HOW IT WORKS SECTION
st.markdown("---")
st.markdown("## 🤔How Does This Work?")

with st.expander("Click to learn about Machine Learning"):
    st.markdown("""
    ### 🧠Machine Learning Explained Simply
    
    **What is Machine Learning?**
    - Think of it like teaching a computer to recognize patterns
    - We show it thousands of examples of people and their salaries
    - The computer learns what factors typically lead to higher salaries
    
    **Our Process:**
    1. **Data Collection**: We used 48,842 real census records
    2. **Pattern Recognition**: The algorithm found relationships between features (age, education, etc.) and income
    3. **Model Training**: We taught 3 different algorithms and picked the best one
    4. **Prediction**: Now it can predict income for new people based on learned patterns
    
    **Key Factors the Model Considers:**
    - Age and work experience
    - Education level
    - Type of work and hours per week
    - Marital status and family situation
    - Geographic and demographic factors
    
    **Accuracy**: Our model is correct about {:.0f}% of the time!
    """.format(model_data['accuracy']*100 if model_data else 75))

with st.expander("Technical Details"):
    if model_data:
        st.markdown(f"""
        ### 🔧Technical Information
        
        **Algorithm**: {model_data['model_name']}
        **Accuracy**: {model_data['accuracy']*100:.2f}%
        **Features**: {len(model_data['feature_names'])}
        
        **Feature List**:
        {', '.join(model_data['feature_names'])}
        
        **Data Preprocessing**:
        - Categorical variables encoded using Label Encoding
        - Numerical features scaled using Standard Scaler
        - Train/test split: 80%/20%
        - Stratified sampling to maintain class balance
        """)
    else:
        st.error("Model data not available")

#FOOTER
st.markdown("---")
st.markdown("### 💡About This Project")
st.markdown("""
This salary prediction app was built as part of an IBM SkillBuild Internship project. 
It demonstrates the complete machine learning pipeline from data analysis to deployment.

**Technologies Used:**
- Python for data science and ML
- Scikit-learn for machine learning algorithms  
- Streamlit for web application framework
- Plotly for interactive visualizations

**Disclaimer**: This model is for educational purposes. Actual salary predictions may vary 
based on many factors not included in this dataset.
""")