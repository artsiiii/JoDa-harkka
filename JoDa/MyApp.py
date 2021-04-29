import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import pydeck as pdk
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import sklearn.metrics as metrics
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV


class interactive():

    # Get data from json
    df = pd.read_json('clearedCarDataWLL2.json')
    
    st.header('Car data')

    # Print dataframe
    st.write(df)  

    counts = pd.DataFrame(df['brand'].value_counts()).reset_index()

    st.header('Cars count by brand')

    st.write(alt.Chart(counts).mark_bar().encode(
        x = alt.X('index', sort=None),
        y = 'brand',
    ))

    # All data averages
    avg_year = int(round(df['year'].mean(),0))
    avg_price = int(round(df['price'].mean(),0))
    avg_kilometer = int(round(df['kilometer'].mean(),0))

    # Dropdowns
    table = df.loc[df['year'] != 0,:]
    
    selected_city = 'All';
    selected_brand = 'All';
    selected_model = 'All';
       
        
    city = df['city'].drop_duplicates().to_numpy()
    city = np.insert(city, 0,'All')
    c = st.empty()
    selected_city = c.selectbox('Pick a city', city)
    
    if selected_city != 'All':
        table = table.loc[table['city'] == selected_city]
    
        
    
    
    car_brands = table['brand'].drop_duplicates().to_numpy()
    car_brands = np.insert(car_brands, 0,'All')
    a = st.empty()
    selected_brand = a.selectbox('Pick a brand', car_brands)
    
       
    if selected_brand != 'All':  
        table = table.loc[table['brand'] == selected_brand]       
        car_models = table['model'].drop_duplicates().to_numpy()
        car_models = np.insert(car_models, 0, 'All')
        b = st.empty()
        selected_model = b.selectbox('Pick a model', car_models)
        
    if selected_model != 'All':
        table = table.loc[table['model'] == selected_model]
            
    
    
    
    
    # Sliders
    min_max_value_price = st.slider('Select min and max price', table['price'].min(), table['price'].max(), [table['price'].min(), table['price'].max()])
    table = table.loc[table['price'] >= min_max_value_price[0] ,:]
    table = table.loc[table['price'] <= min_max_value_price[1] ,:]
    
    min_max_value_kilometer = st.slider('Select min and max kilometers', table['kilometer'].min(), table['kilometer'].max(), [table['kilometer'].min(), table['kilometer'].max()])
    table = table.loc[table['kilometer'] >= min_max_value_kilometer[0] ,:]
    table = table.loc[table['kilometer'] <= min_max_value_kilometer[1] ,:]
    
   
    min_max_value_year = st.slider('Select min and max year', table['year'].min(), table['year'].max(), [table['year'].min(), table['year'].max()])
    table = table.loc[table['year'] >= min_max_value_year[0] ,:]
    table = table.loc[table['year'] <= min_max_value_year[1] ,:]
    
    min_max_value_latitude = st.slider('Select min and max latitude', table['lat'].min(), table['lat'].max(), [table['lat'].min(), table['lat'].max()])
    table = table.loc[table['lat'] >= min_max_value_latitude[0] ,:]
    table = table.loc[table['lat'] <= min_max_value_latitude[1] ,:]
    
    min_max_value_longitude = st.slider('Select min and max longitude', table['lon'].min(), table['lon'].max(), [table['lon'].min(), table['lon'].max()])
    table = table.loc[table['lon'] >= min_max_value_longitude[0] ,:]
    table = table.loc[table['lon'] <= min_max_value_longitude[1] ,:]
    
    
    
    # Selected data averages
    avg_year_s = int(round(table['year'].mean(),0))
    avg_price_s = int(round(table['price'].mean(),0))
    avg_kilometer_s = int(round(table['kilometer'].mean(),0))

    st.text('Total count by parameters: ' + str(table.shape[0]))
    st.text('Average manufacturing year by parameters: ' + str(avg_year_s))
    st.text('Average price by parameters: ' + str(avg_price_s) + ' €')
    st.text('Average kilometers by parameters: ' + str(avg_kilometer_s) + ' km')
    
    # Alt chart
    y_ax = st.radio('Select y-axis', ['price', 'kilometer'])

    chart = alt.Chart(table).mark_circle().encode(
        x = alt.X('year:N'),
        y = alt.Y(y_ax),
        tooltip=['brand', 'model', 'price' ,'year', 'kilometer','city']).properties(
        width=500,
        height=500
    )
    st.altair_chart(chart, use_container_width=True)
    
    # Selected table
    st.write(table)
    
    

    # Map
    co = pd.DataFrame(table['lat'])
    co['lon'] = table['lon']
    st.map(co)
    
    # Scatter matrix
    pd.plotting.scatter_matrix(table, alpha = 0.5, figsize = (10,10))
    st.pyplot(plt)
    
    
    # X values for linear regression
    #st.write('Select X variables for linear regression')
    
    featuresOption = st.multiselect('Select X variables:', ['brand','model','price','year','kilometer','city','lon','lat'])
    features = featuresOption
    dumFeatures = []
    
    if 'brand' in features:
        dumFeatures.append('brand')
        features.remove('brand')
        
    if 'model' in features:
        dumFeatures.append('model')
        features.remove('model')
                
    if 'city' in features:
        dumFeatures.append('city')
        features.remove('city')
    
    
        
    y_variable = st.radio('Select y variable for linear regression', ['brand','model','price','year','kilometer','city','lon','lat'])
    
    
    X = table[features]
    
    if len(dumFeatures) != 0:
        X = pd.concat([X,pd.get_dummies(table[dumFeatures])], axis = 1)
        
   
    if 'brand' in y_variable:
        y = pd.get_dummies(table[['brand']])        
    elif 'model' in y_variable:
        y = pd.get_dummies(table[['model']])                
    elif 'city' in y_variable:
        y = pd.get_dummies(table[['city']])
    else:
        y = table[y_variable]
    
    
    
    buttonModel = st.button('Start modeling')
    clf = None
    
    if buttonModel:    
        
        # Model learning
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        clf = LinearRegression().fit(X_train,y_train)
        st.write('Score:')
        st.write(clf.score(X_test,y_test))
                       
    
        # Predicting
        st.write("Predicted value: ")
        st.write(clf.predict(X_test.iloc[-1,:].values.reshape(1,len(X_test.columns)))[0])
        st.write("Correct value: ")
        st.write(y_test.iloc[-1])
        
        
            
        

        


        

       
    
    
