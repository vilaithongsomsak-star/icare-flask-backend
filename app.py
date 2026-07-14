"""
iCare Benefit — Flask ML Backend
Thesis: Predicting Profitability using Machine Learning
NUOL M.Sc. Computer Science (FinTech)
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://predictprofit.netlify.app"}})

BASE = os.path.dirname(__file__)

# Load
with open(os.path.join(BASE,'model_lr.pkl'),'rb') as f:
    lr_model, scaler = pickle.load(f)
with open(os.path.join(BASE,'model_rf.pkl'),'rb') as f:
    rf_model = pickle.load(f)
with open(os.path.join(BASE,'model_xgb_best.pkl'),'rb') as f:
    xgb_model = pickle.load(f)

df_hist = pd.read_csv(os.path.join(BASE, 'icare_clean.csv'))

FEATURE_LIST = [
    'revenue','cost_of_sales','op_exp','other_exp','gross_profit',
    'gross_margin','cost_ratio','op_exp_ratio','other_exp_ratio',
    'total_cost_ratio','rev_per_member','rev_per_order',
    'month','quarter','is_q4','month_sin','month_cos',
    'revenue_lag1','netprofit_lag1','cost_lag1','gmargin_lag1',
    'revenue_roll3','cost_roll3','profit_roll3',
    'revenue_growth','cost_growth',
    'sales_orders','icare_members',
]

def build_features(data: dict) -> pd.DataFrame:
    rev   = float(data['revenue'])
    cos   = float(data['cost_of_sales'])
    op    = float(data['op_exp'])
    other = float(data['other_exp'])
    month = int(data.get('month', 6))
    so    = float(data.get('sales_orders', 250))
    mem   = float(data.get('icare_members', 90))

    gp = rev - cos
    gm = gp / rev if rev else 0
    cr = cos / rev if rev else 0
    opr = op / rev if rev else 0
    othr = other / rev if rev else 0
    tcr  = (cos + op + other) / rev if rev else 0
    npm  = (rev - cos - op - other) / rev if rev else 0
    rpm  = rev / mem if mem else 0
    rpo  = rev / so  if so  else 0

    qtr  = ((month - 1) // 3) + 1
    is_q4 = 1 if qtr == 4 else 0
    msin = np.sin(2 * np.pi * month / 12)
    mcos = np.cos(2 * np.pi * month / 12)

    # Lag/Rolling: use historical medians as defaults
    rev_lag1   = float(df_hist['revenue'].median())
    np_lag1    = float(df_hist['net_profit'].median())
    cost_lag1  = float(df_hist['cost_of_sales'].median())
    gm_lag1    = float(df_hist['gross_margin'].median())
    rev_roll3  = float(df_hist['revenue'].median())
    cost_roll3 = float(df_hist['cost_of_sales'].median())
    pr_roll3   = float(df_hist['net_profit'].median())
    rev_growth = float(df_hist['revenue_growth'].median())
    cost_growth= float(df_hist['cost_growth'].median())

    row = {
        'revenue': rev, 'cost_of_sales': cos, 'op_exp': op, 'other_exp': other,
        'gross_profit': gp, 'gross_margin': gm, 'cost_ratio': cr,
        'op_exp_ratio': opr, 'other_exp_ratio': othr, 'total_cost_ratio': tcr,
        'rev_per_member': rpm, 'rev_per_order': rpo,
        'month': month, 'quarter': qtr, 'is_q4': is_q4,
        'month_sin': msin, 'month_cos': mcos,
        'revenue_lag1': rev_lag1, 'netprofit_lag1': np_lag1,
        'cost_lag1': cost_lag1, 'gmargin_lag1': gm_lag1,
        'revenue_roll3': rev_roll3, 'cost_roll3': cost_roll3,
        'profit_roll3': pr_roll3, 'revenue_growth': rev_growth,
        'cost_growth': cost_growth, 'sales_orders': so, 'icare_members': mem,
    }
    return pd.DataFrame([row])[FEATURE_LIST]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        X = build_features(data)
        X_s = scaler.transform(X)

        pred_lr  = float(lr_model.predict(X_s)[0])
        pred_rf  = float(rf_model.predict(X)[0])
        pred_xgb = float(xgb_model.predict(X)[0])

        rev = float(data['revenue'])
        cos = float(data['cost_of_sales'])
        op  = float(data['op_exp'])
        oth = float(data['other_exp'])
        gp  = rev - cos
        actual_net = rev - cos - op - oth

        return jsonify({
            'status': 'success',
            'input': {
                'revenue': rev, 'cost_of_sales': cos,
                'op_exp': op, 'other_exp': oth,
                'gross_profit': round(gp, 2),
                'actual_net_profit': round(actual_net, 2),
            },
            'predictions': {
                'linear_regression': round(pred_lr, 2),
                'random_forest':     round(pred_rf, 2),
                'xgboost':           round(pred_xgb, 2),
            },
            'metrics': {
                'LR':  {'R2': -5.25, 'MAE': 17191, 'RMSE': 23736},
                'RF':  {'R2': 0.43,  'MAE': 6331,  'RMSE': 7197},
                'XGB': {'R2': 0.54,  'MAE': 5936,  'RMSE': 6446},
            },
            'best_model': 'XGBoost',
            'gross_margin': round((gp/rev)*100, 2) if rev else 0,
            'net_margin':   round((actual_net/rev)*100, 2) if rev else 0,
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/history', methods=['GET'])
def history():
    records = df_hist[['period','revenue','cost_of_sales',
                        'gross_profit','net_profit','gross_margin',
                        'net_profit_margin']].tail(12).to_dict(orient='records')
    return jsonify({'status':'success','data': records})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status':'ok','models':['LR','RF','XGBoost']})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
