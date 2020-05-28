from flask import Flask, render_template, request
import pandas as pd
import tensorflow as tf
import os
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from ml_model import metric_json
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model('models/prediction_model.h5', compile=True)

data_url = url = 'https://raw.github.com/buraksekili/airbnb-price-prediction/master/data/unlabeled-data.csv'
data = pd.read_csv(url)

neighbourhood_coordinates = {
    "Manhattan": (40.783058, -73.971252),
    "Brooklyn": (40.678177, -73.944160),
    "Queens": (40.728226, -73.794853),
    "Bronx": (40.844784, -73.864830),
    "Staten Island": (40.579533, -74.150200)
}


def get_prediction(model_, input_df):
    try:
        input_df = tf.convert_to_tensor(input_df, dtype=tf.float64)
    except Exception as e:
        print(e)

    prediction = model_.predict(input_df)
    prediction = float("{:.2f}".format(prediction[0][0]))
    return prediction


def label_all(input_df):
    categorical = input_df.select_dtypes(include=['object']).columns
    for i in categorical:
        input_df[i] = LabelEncoder().fit_transform(input_df[i])


@app.route("/", methods=['POST', 'GET'])
def main_page():
    return render_template("index.html")


@app.route("/result", methods=['POST', 'GET'])
def display_result():
    prediction_global = 0
    if request.method == "POST":

        metric_json['neighbourhood_group'] = request.form['neighbourhood_group']
        metric_json['room_type'] = request.form['room_type']
        metric_json['accommodates'] = request.form['accommodates']
        metric_json['latitude'] = neighbourhood_coordinates[metric_json['neighbourhood_group']][0]
        metric_json['longitude'] = neighbourhood_coordinates[metric_json['neighbourhood_group']][1]

        if request.form['bathrooms'] is not '':
            metric_json['bathrooms'] = request.form['bathrooms']

        if request.form['bedrooms'] is not '':
            metric_json['bedrooms'] = (request.form['bedrooms'])

        if request.form['beds'] is not '':
            metric_json['beds'] = request.form['beds']

        input_raw_df = pd.DataFrame.from_dict(metric_json, orient='index')

        input_raw_df = input_raw_df.T

        for i in input_raw_df.columns:
            try:
                input_raw_df[i] = float(input_raw_df[i])
            except ValueError:
                pass

        all_raw_data = pd.concat([data, input_raw_df])
        all_raw_data.drop(columns=['Unnamed: 5'], inplace=True)
        label_all(all_raw_data)

        scaler = StandardScaler()
        all_columns = all_raw_data.columns
        all_raw_data[all_columns] = scaler.fit_transform(all_raw_data[all_columns])
        labeled_input = all_raw_data.iloc[[-1]]

        prediction_global = get_prediction(model, labeled_input)

    return render_template("index.html", result_price=prediction_global,
                           neighbourhood_group_name=metric_json['neighbourhood_group'],
                           room_type=metric_json['room_type'],
                           accommodates=metric_json['accommodates'],
                           bathrooms=metric_json['bathrooms'],
                           bedrooms=metric_json['bedrooms'],
                           beds=metric_json['beds'])


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
