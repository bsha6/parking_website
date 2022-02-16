from cmath import log
from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

# blue text on home page from css class

# fp = '/Users/blake/Documents/TDI/parking_violations/Data/'
dir = os.path.dirname(__file__)
fp = os.path.join(dir, 'model/')

# log_reg_model = pickle.load(open(fp + 'log_reg_5_feats.pkl', 'rb'))
xgb_model = pickle.load(open(fp + '6_feats_xgb_final.pkl', 'rb'))
pred_key = {0: 'Denied', 1.0: 'Reduced', 2.0: 'Granted'}

# print(xgb_model.get_feature_names_out)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def ticket_fields():
    license_types_list = sorted(['999','OMT', 'OMS', 'VAS', 'LMB', 'AYG', 'LMA',
       'TRC', 'ITP', 'OMR', 'MED', 'CMB', 'SOS', 'IRP', 'APP', 'NYC',
       'ORG', 'SPO', 'PSD', 'LMC', 'SRN', 'TRL', 'DLR', 'MCL', 'TOW',
       'SCL', 'AMB', 'RGL', 'TRA', 'ORC', 'PHS', 'CHC', 'NLM', 'HIS',
       'SPC', 'STA', 'OML', 'BOB', 'AGR', 'MCD', 'SNO', 'RGC', 'BOT',
       'SEM', 'NYS', 'JCL', 'USC', 'GSM', 'ATV', 'NYA', 'LTR', 'HAM',
       'STG', 'CSP', 'AGC', 'HOU',  'ORM', 'MOT', 'COM', 'SRF', 'PAS'])
    # vehicle_body_types_list = sorted(['TR', 'P/SH', 'TT', 'OM', 'W/DR', 'SUV', 'TWOD', 'AMBU', '5D', 'CMIX', '00', 'LIM', 'O', 'MTR', 'LTRL', 'YY', 'CH', 'H/WH', 'SEMI', 'TRC', '4W', 
    #                           'TRAV', '2S', 'TR/E', '4H', 'CV', 'WG', 'TRAI', 'MC', 'BOAT', 'TOW', 'TANK', 'T/CR', 'TRK', 'LL', 'VN', 'MCC', 'ST', 'MOPD', 'MOBL', 'MP', 'PV', 
    #                           'TK', 'CP', 'HB', 'WAGO', 'STAK', 'CG', 'BUS', 'DUMP', 'SWT', 'TAXI', 'TR/C', 'FLAT', 'CONV', 'TRLR', 'UTIL', '2DSD', 'MCY', 'SEDN', 'TRAC', 'REFG', 
    #                           'PICK', 'DELV', '4DSD', 'SUBN', 'VAN'])
    vbt_dict = {'2DSD': 'TWO-DOOR SEDAN', '4DSD': 'FOUR-DOOR SEDAN', 'AMBU': 'AMBULANCE', 'ATV': 'ALL TERRAIN VEHICLE', 'BOAT': 'BOAT', 'BUS': 'BUS(OMNIBUS)', 'CMIX': 'CEMENT MIXER', 
                'CONV': 'CONVERTIBLE', 'CUST': 'CUSTOM', 'DCOM': 'DISABLED COMMERICAL', 'DELV': 'DELIVERY TRUCK', 'DUMP': 'DUMP TRUCK', 'EMVR': 'EARTH MOVER', 'FIRE': 'FIRE TRUCK', 
                'FLAT': 'FLAT BED TRUCK', 'FPM': 'FEED PROCESSING MACHINE', 'H/IN': 'HEARSE-INVALID', 'H/TR': 'HOUSE TRAILER', 'H/WH': 'HOUSE ON WHEELS', 'HRSE': 'HEARSE(AMBULANCE)',
                'LIM': 'LIMOUSINE(OMNIBUS)', 'LOCO': 'LOCOMOTIVE', 'LSV': 'LOW SPEED VEHICLE', 'LSVT': 'LOW SPEED VEHICLE - TRUCK', 'LTRL': 'LIGHT TRAILER', 'MCC': 'MOBILE CAR CRUSHER', 
                'MCY': 'MOTORCYCLE', 'MFH': 'MANUFACTURED HOME', 'MOBL': 'SNOWMOBILE', 'MOPD': 'MOPED', 'N/A': 'NOT APPLICABLE', 'P/SH': 'POWER SHOVEL', 'PICK': 'PICK-UP TRUCK', 
                'POLE': 'POLE TRAILER', 'R/RD': 'ROAD ROLLER', 'RBM': 'ROAD BUILDING MACHINE', 'RD/S': 'ROAD SWEEPER', 'REFG': 'REFRIGERATOR TRAILER', 'RPLC': 'REPLICA',
                'S/SP': 'SAND OR AGRICULTRAL SPREADER/SPRAYER', 'SEDN': 'SEDAN', 'SEMI': 'SEMI-TRAILER', 'SN/P': 'SNOW PLOW', 'SNOW': 'SNOWMOBILE', 'STAK': 'STAKE TRUCK', 'SUBN': 'SUBURBAN', 
                'SWT': 'TRUCK W/SMALL WHEELS', 'T/CR': 'TRACTOR CRANE', 'TANK': 'TANK TRUCK', 'TAXI': 'TAXI', 'TOW': 'TOW TRUCK', 'TR/C': 'TRUCK CRANE', 'TR/E': 'TRACTION ENGINE', 
                'TRAC': 'TRACTOR', 'TRAV': 'SNOW TRAVELER', 'TRLR': 'TRAILER', 'UTIL': 'UTILITY', 'VAN': 'VAN', 'W/DR': 'WELL DRILLER', 'W/SR': 'WELL SERVICING RIG'}
    return render_template('index.html', license_types_list=license_types_list, vbt_dict=vbt_dict, current_pg='home')

@app.route("/submission", methods=["POST"])
def submit():
    fine = request.form["fine-amount"]
    license_type = request.form["license_type"]
    vehicle_body_type = request.form["vehicle_body_type"]
    vehicle_color = request.form["vehicle-color"]
    vehicle_make = request.form["vehicle-make"].upper()
    viol_code = request.form["violation-code"]

    if len(fine) >= 1:
        fine = float(fine)
    else:
        fine = 115.0

    # Convert to df (use dict?)
    submitted_dict = {'fine_amount': fine, 'license_type': license_type, 'vehicle_body_type_cleaned': vehicle_body_type,
                      'vehicle_color_cleaned': vehicle_color, 'vehicle_make': vehicle_make, 'violation_code': viol_code}
                      
    df_submitted = pd.DataFrame(submitted_dict, index=[0])

    # Calculate probabilities
    pred = xgb_model.predict(df_submitted)
    pred_prob = xgb_model.predict_proba(df_submitted)[0]

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

@app.route("/next_steps", methods=["GET"])
def next_steps():
    return render_template('next_steps.html', current_pg='next_steps')

if __name__ == "__main__":
    app.run()

# divs that are each labelled as row/col