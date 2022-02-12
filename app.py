from cmath import log
from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

# bootstrap, jinja 2 template
# clean up vehicle make feature
# improve model (ensemble)

# how to get each page highlighted... modify class tag in base?
# Why is text on home pg blue?

fp = '/Users/blake/Documents/TDI/parking_violations/Data/'
log_reg_model = pickle.load(open(fp + 'log_reg_5_feats.pkl', 'rb'))
pred_key = {0: 'Denied', 1.0: 'Reduced', 2.0: 'Granted'}

app = Flask(__name__)

@app.route("/", methods=["GET"])
def ticket_info():
    counties_list = ['NY', 'K', 'Q', 'BX', 'BK', 'MN', 'QN', 'R', 'ST', 'None']
    return render_template('index.html', counties_list=counties_list, current_pg='home')

@app.route("/submission", methods=["POST"])
def submit():
    fine = float(request.form["fine-amount"])
    viol_code = request.form["violation-code"]
    county = request.form["county"]
    issuer_squad = request.form["issuer-squad"]
    vehicle = request.form["vehicle-make"].upper()

    # Convert to df (use dict?)
    submitted_dict = {'violation_code': viol_code, 'county': county, 
                      'issuer_squad': issuer_squad, 'vehicle_make': vehicle, 
                      'fine_amount': fine}
    df_submitted = pd.DataFrame(submitted_dict, index=[0])

    # Calculate probabilities
    pred = log_reg_model.predict(df_submitted)
    pred_prob = log_reg_model.predict_proba(df_submitted)[0]
    denied_pred_prob = pred_prob[0]
    granted_reduced_pred_prob = pred_prob[1] + pred_prob[2]
    diff_pred_prob = abs(denied_pred_prob - granted_reduced_pred_prob)
    print(diff_pred_prob, denied_pred_prob, granted_reduced_pred_prob)

    if diff_pred_prob < 0.1:
        confidence_level = 'low'
    elif 0.1 <= diff_pred_prob < 0.3:
        confidence_level = "medium"
    elif 0.3 <= diff_pred_prob:
        confidence_level = "high"
    else:
        confidence_level = "low"

    pred_txt = pred_key[pred[0]]
    output = f"Fine: {fine}, Violation Code: {viol_code}, County: {county}, \
    #     Issuer Squad: {issuer_squad}, Vehicle: {vehicle}, Prediction: {pred_txt}"

    # if pred_txt == 'Granted' or pred_txt == 'Reduced':
    #     granted_output = f"You should dispute! Our model predicts a {confidence_level} chance of your appeal being granted or fine amount being reduced. {pred_prob}"
    #     return granted_output
    # # elif pred_txt == 'Reduced':
    # #     return output
    # elif pred_txt == 'Denied':
    #     denied_output = f"Unfortunately, it's unlikely your dispute will be successful. Our model gives it a {confidence_level} chance of being denied. {pred_prob}"
    #     return denied_output

    # Format probabilities for website
    denied_prob_output = "{0:.1f}%".format(denied_pred_prob*100)
    reduced_prob_output = "{0:.1f}%".format(pred_prob[1]*100)
    granted_prob_output = "{0:.1f}%".format(pred_prob[2]*100)

    return render_template('rec.html', denied_prob=denied_prob_output, reduced_prob=reduced_prob_output, granted_prob=granted_prob_output, 
                          confidence_level=confidence_level, pred_txt=pred_txt)

@app.route("/how_it_works", methods=["GET"])
def how():
    return render_template('how_it_works.html', current_pg='how_it_works')

@app.route("/contact", methods=["GET"])
def contact():
    return render_template('contact.html', current_pg='contact')

if __name__ == "__main__":
    app.run(debug=True)

# divs that are each labelled as row/col