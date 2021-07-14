from flask import Flask, request, render_template
import numpy as np
from my_utili import my_function, positive, negative, final_labels

app = Flask(__name__)

# Bind home function to URL
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/disclaimer')
def disclaimer():
    return render_template('disclaimer.html')


# Bind predict function to URL
@app.route('/predict', methods =['POST'])
def predict():
    # Put all form entries values in a list 
    features = request.form.getlist("Tumor")

    all_features = ['solid', 'solid_necrosis','cystic','unilocular','honeycomb','vegetations', 'papillary','multi','hypointese','calcification','haemo','dark','fat','diffusion','endo']
    postive_features = [1 if feature in features else 0 for feature in all_features]
    postive_features = np.array(postive_features).reshape(15,1)

    result1 =  my_function(pos_weights=positive, neg_weights=negative, some_pos_row=postive_features, labels=final_labels)

    
    # Check the output values and retrive the result with html tag based on the value
    if len(features) != 0:
        return render_template('index.html', 
                               result =  result1[:10])

    else:
        return render_template('index.html', 
                               result = '')

if __name__ == '__main__':
#Run the application
    app.run()