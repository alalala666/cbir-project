from flask import Flask, render_template, request, jsonify
import os

# create flask instance
app = Flask(__name__)

# main route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html')

# run!
if __name__ == '__main__':
    app.run('0.0.0.0', debug=True)