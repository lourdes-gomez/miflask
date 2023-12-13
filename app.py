from markupsafe import escape
from flask import Flask, jsonify, request, render_template, url_for, flash, redirect
from sqlalchemy import create_engine, text
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import json 
import sqlite3
from io import BytesIO
import base64
import matplotlib.pyplot as plt


file_path = r'c:\\Users\\lour2\\Desktop\\LOURDES\\data science\\carpeta_trabajo\\repo_sept_23\\4-Data_Engineering\\Mini proyecto\\pipeline.pkl'
with open(file_path, 'rb') as archivo_entrada:    #rb 'read bytes'
    xgbmodel = pickle.load(archivo_entrada)


#engine = create_engine('sqlite:///c:\\Users\\lour2\\Desktop\\LOURDES\\data science\\carpeta_trabajo\\repo_sept_23\\4-Data_Engineering\\Mini proyecto\\predictions.db')
connection = sqlite3.connect("postgresql://fl0user:bd1sikXoSr0Y@ep-icy-sunset-15884403.ap-southeast-1.aws.neon.fl0.io:5432/database?sslmode=require", check_same_thread=False) #conexion con la base de datos


app = Flask(__name__)
app.config['SECRET_KEY'] = '44360ee74d98f3b625851f58d58e73e3d58089bf15055234'



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', messages= "Prediction of physical condition")


#1.Endpoint para predecir
@app.route('/api/v0/predict', methods=('GET', 'POST'))
def predict():
    if request.method == 'POST':
        edad = request.form['Edad']
        genero = request.form['Genero']
        altura = request.form['Altura']
        peso = request.form['Peso']
        grasa = request.form['Grasa']
        presion_dia = request.form['Presión_diastólica']
        presion_sis = request.form['Presión_sistólica']
        flex = request.form['Flexibilidad']

        if not edad:
            flash('Edad is required!')
        elif not genero:
            flash('Genero is required!')
        elif not altura:
            flash('Altura is required!')
        elif not peso:
            flash('Peso is required!')
        elif not grasa:
            flash('Grasa is required!')
        elif not presion_dia:
            flash('Presión is required!')
        elif not presion_sis:
            flash('Presión is required!')
        elif not flex:
            flash('Flexibilidad is required!')


        else:
            datos_cliente = [[edad,genero,altura,peso,grasa,presion_dia,presion_sis,flex]]
            #predice
            predictions = xgbmodel.predict(datos_cliente)  

            #meter fecha , inputs y predicciones en un diccionario
            pred_db = pd.DataFrame({
            'Fecha': datetime.strftime(datetime.now(), '%Y-%m-%d'),
            'Inputs': [[edad,genero,altura,peso,grasa,presion_dia,presion_sis,flex]],
            'Predictions': predictions  })
            pred_db['Inputs'] = pred_db['Inputs'].apply(json.dumps)
            #lanza el diccionario a la base de datos
            pred_db.to_sql('logs_predictions', con=connection, if_exists='append', index=False)
            return 'Su condición física es: '+ str(predictions)
            
    return render_template('predict.html')

    

#1.Endpoint para consultar logs
@app.route('/api/v0/consult', methods=['GET','POST'])
def consult(): 

    if request.method == 'POST':
        query = """SELECT * FROM logs_predictions"""
        data = pd.read_sql_query(query, connection)
        
        data['Fecha']= pd.to_datetime(data['Fecha'], format='mixed ')
        results = []
        
        start = request.form['start']
        start = datetime.strptime(start, '%Y-%m-%d')
        end = request.form['end']
        end = datetime.strptime(end, '%Y-%m-%d')
            
        mask = (data['Fecha'] >= start) & (data['Fecha'] <= end)
        results = data.loc[mask].to_dict(orient='records')

        return jsonify(results)
        
    return render_template('logs.html')

@app.route('/api/v0/feature_importance', methods=['GET'])
def generate_plot():    
    feature_importances = xgbmodel.named_steps['xgb'].feature_importances_
    fi = pd.DataFrame({'Features' : ['age', 'gender', 'height_cm', 'weight_kg', 'body fat_%', 'diastolic',
       'systolic', 'sit and bend forward_cm'], 
              'Importance' : feature_importances}).sort_values(by= 'Importance', ascending = False)
    plt.bar(fi['Features'], fi['Importance'])
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    
    # Encode the image as base64 for embedding in HTML
    plot_data = base64.b64encode(image_stream.read()).decode('utf-8')
    
    plt.close()  # Close the plot to free up resources

    return render_template('feature_importance.html', plot_data=plot_data)

@app.route('/About/', methods=['GET'])
def About():
    
    return render_template('about.html')


if __name__ == '__main__':   
    app.run(debug = True, port = 8080)