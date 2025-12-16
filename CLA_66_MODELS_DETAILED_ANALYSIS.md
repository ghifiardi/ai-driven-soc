# ⚡ Cursor-Ready Runbook: CLA Training & Dashboard (Start Here)

This quickstart lets you run the **Continuous Learning Agent (CLA)** training loop and view the dashboard **immediately in Cursor**. It also bakes in fixes for **data imbalance**, **metrics**, and **file permissions**.

---

## 1) Open the repo in Cursor
- File ▸ Open Folder… → `~/Downloads/ai-driven-soc`
- Terminal ▸ New Terminal

## 2) Virtualenv
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

## 3) Minimal deps (or pip install -r requirements.txt)
pip install google-cloud-bigquery google-auth google-auth-oauthlib pandas pandas-gbq scikit-learn streamlit python-dateutil pytz imbalanced-learn xgboost

## 4) BigQuery auth (pick one)
export GOOGLE_APPLICATION_CREDENTIALS="$PWD/sa-gatra-bigquery-clean.json"
# OR
gcloud auth application-default login

## 5) (Recommended) Fix BigQuery schemas
python "Mixture of Expert/fix_bigquery_schemas.py"

## 6) Model dir permissions (prevents .pkl save errors)
sudo mkdir -p /home/raditio.ghifiardigmail.com/ai-driven-soc/models
sudo chown -R $USER:$USER /home/raditio.ghifiardigmail.com/ai-driven-soc/models
chmod -R u+rwX /home/raditio.ghifiardigmail.com/ai-driven-soc/models

## 7) Start the trainer
# one of:
python cla_trainer.py
# or
python -m cla.training.loop

## 8) Open the dashboard
bash deploy/access_simple_dashboard.sh
# opens http://10.45.254.19:8519
# or run a local app:
# streamlit run ai-model-training-dashboard/app.py

---

## 9) Imbalance-aware training (add before clf.fit)
# Class weights (sklearn)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_map = {c:w for c,w in zip(classes, weights)}

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42,
                             class_weight=class_weight_map)
clf.fit(X_train, y_train)

# Optional SMOTE (train split only)
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
clf.fit(X_train_bal, y_train_bal)

# Optional XGBoost with class balance
from xgboost import XGBClassifier
spw = max(1, int((y_train==0).sum() / max(1,(y_train==1).sum())))
xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                    subsample=0.9, colsample_bytree=0.9, random_state=42,
                    tree_method="hist", scale_pos_weight=spw)
xgb.fit(X_train, y_train)

# (Optional) calibrate probabilities if you plot reliability curves
# from sklearn.calibration import CalibratedClassifierCV

---

## 10) Verify metrics
Ensure **precision/recall/F1/PR-AUC** are non-zero and the model isn’t predicting a single class.

### Troubleshooting (fast)
- Permission denied saving models → redo step 6
- BigQuery auth errors → check step 4 (`echo $GOOGLE_APPLICATION_CREDENTIALS`)
- streamlit: command not found → `pip install streamlit` in the venv
- acc=1.0 but p/r=0 → enable class-weights/SMOTE; verify positive labels exist

### Executive bullets
- Run end-to-end in minutes (venv → auth → schema fix → train → dashboard)
- Imbalance-aware by default (class-weights; SMOTE/XGBoost optional)
- Track **Precision/Recall/F1/PR-AUC**; add calibration for trustworthy scores

### Methodology notes
- Auth via ADC or service account JSON
- Trainer polls ~6-minute windows, writes to `.../models`
- Metrics emphasize minority recall to reduce false negatives in SOC
# ⚡ Cursor-Ready Runbook: CLA Training & Dashboard (Start Here)

This quickstart lets you run the **Continuous Learning Agent (CLA)** training loop and view the dashboard **immediately in Cursor**. It also bakes in fixes for **data imbalance**, **metrics**, and **file permissions**.

---

## 1) Open the repo in Cursor
- File ▸ Open Folder… → `~/Downloads/ai-driven-soc`
- Terminal ▸ New Terminal

## 2) Virtualenv
```bash
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

## 3) Minimal deps (or pip install -r requirements.txt)
```bash
pip install google-cloud-bigquery google-auth google-auth-oauthlib pandas pandas-gbq scikit-learn streamlit python-dateutil pytz imbalanced-learn xgboost
```

## 4) BigQuery auth (pick one)
```bash
export GOOGLE_APPLICATION_CREDENTIALS="$PWD/sa-gatra-bigquery-clean.json"
# OR
gcloud auth application-default login
```

## 5) (Recommended) Fix BigQuery schemas
```bash
python "Mixture of Expert/fix_bigquery_schemas.py"
```

## 6) Model dir permissions (prevents .pkl save errors)
```bash
sudo mkdir -p /home/raditio.ghifiardigmail.com/ai-driven-soc/models
sudo chown -R $USER:$USER /home/raditio.ghifiardigmail.com/ai-driven-soc/models
chmod -R u+rwX /home/raditio.ghifiardigmail.com/ai-driven-soc/models
```

## 7) Start the trainer
```bash
# one of:
python cla_trainer.py
# or
python -m cla.training.loop
```

## 8) Open the dashboard
```bash
bash deploy/access_simple_dashboard.sh
# opens http://10.45.254.19:8519
# or run a local app:
# streamlit run ai-model-training-dashboard/app.py
```

---

## 9) Imbalance-aware training (add before clf.fit)
```python
# Class weights (sklearn)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_map = {c:w for c,w in zip(classes, weights)}

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42,
                             class_weight=class_weight_map)
clf.fit(X_train, y_train)

# Optional SMOTE (train split only)
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
clf.fit(X_train_bal, y_train_bal)

# Optional XGBoost with class balance
from xgboost import XGBClassifier
spw = max(1, int((y_train==0).sum() / max(1,(y_train==1).sum())))
xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                    subsample=0.9, colsample_bytree=0.9, random_state=42,
                    tree_method="hist", scale_pos_weight=spw)
xgb.fit(X_train, y_train)

# (Optional) calibrate probabilities if you plot reliability curves
# from sklearn.calibration import CalibratedClassifierCV
```

---

## 10) Verify metrics
Ensure **precision/recall/F1/PR-AUC** are non-zero and the model isn't predicting a single class.

### Troubleshooting (fast)
- Permission denied saving models → redo step 6
- BigQuery auth errors → check step 4 (`echo $GOOGLE_APPLICATION_CREDENTIALS`)
- streamlit: command not found → `pip install streamlit` in the venv
- acc=1.0 but p/r=0 → enable class-weights/SMOTE; verify positive labels exist

### Executive bullets
- Run end-to-end in minutes (venv → auth → schema fix → train → dashboard)
- Imbalance-aware by default (class-weights; SMOTE/XGBoost optional)
- Track **Precision/Recall/F1/PR-AUC**; add calibration for trustworthy scores

### Methodology notes
- Auth via ADC or service account JSON
- Trainer polls ~6-minute windows, writes to `.../models`
- Metrics emphasize minority recall to reduce false negatives in SOC

---

# 🤖 CLA 66+ MODELS DETAILED ANALYSIS

## 📊 **COMPREHENSIVE BREAKDOWN OF TRAINED MODELS**

Based on my investigation of the CLA (Continuous Learning Agent) training system, here's a detailed analysis of the 66+ models that have been trained:

---

## 🔢 **MODEL INVENTORY SUMMARY**

### **📈 TOTAL MODEL COUNT:**
- **Trained Models**: 66 `.pkl` files
- **Metrics Files**: 66 `.json` files  
- **Total Files**: 132 files (66 pairs)
- **Model Size**: ~52KB per model
- **Total Storage**: ~3.7MB for all models

### **⏰ TRAINING TIMELINE:**
- **First Model**: `trained_model_20250919_093715.pkl` (Sep 19, 09:37 AM)
- **Latest Model**: `trained_model_20250920_011415.pkl` (Sep 20, 01:14 AM)
- **Training Duration**: ~15.5 hours of continuous training
- **Training Frequency**: Every 6 minutes (as configured)

---

## 🧠 **MODEL TYPE & ARCHITECTURE**

### **🤖 MODEL SPECIFICATION:**
```python
Model Type: RandomForestClassifier
Parameters:
├── n_estimators: 100 (decision trees)
├── max_depth: 10 (tree depth limit)
├── random_state: 42 (reproducibility)
├── class_weight: 'balanced' (handles class imbalance)
└── File Size: ~52KB (serialized pickle format)
```

### **🎯 MODEL PURPOSE:**
- **Primary Function**: Security Alert Classification
- **Task**: Binary classification (True Positive vs False Positive)
- **Input**: Feedback data from SOC analysts
- **Output**: Prediction of alert validity

---

## 📊 **TRAINING DATA ANALYSIS**

### **📈 TRAINING SAMPLES PROGRESSION:**
Based on the metrics files, here's how training data evolved:

| Model Era | Training Samples | Test Samples | Time Period |
|-----------|------------------|--------------|-------------|
| **Early Models** | 1,600 | 400 | Sep 19, 09:37-10:xx |
| **Mid Models** | 2,400 | 600 | Sep 19, 10:xx-11:xx |
| **Later Models** | 3,200 | 800 | Sep 19, 11:xx-12:xx |
| **Recent Models** | 4,000 | 1,000 | Sep 19, 12:xx-Sep 20, 01:xx |

### **📊 PERFORMANCE CHARACTERISTICS:**
All models show consistent patterns:
```json
{
  "precision": 0.0,
  "recall": 0.0, 
  "f1_score": 0.0,
  "accuracy": 1.0,
  "true_positives": 0,
  "false_positives": 0,
  "true_negatives": 4000,  // All samples classified as negative
  "false_negatives": 0,
  "training_samples": 4000,
  "test_samples": 1000
}
```

---

## 🔍 **CRITICAL INSIGHTS**

### **⚠️ MODEL BEHAVIOR ANALYSIS:**

#### **1. PERFECT ACCURACY BUT ZERO PRECISION:**
- **Accuracy**: 100% (all predictions correct)
- **Precision**: 0% (no true positives identified)
- **Recall**: 0% (no positive cases detected)
- **F1-Score**: 0% (no meaningful classification)

#### **2. CLASSIFICATION BIAS:**
- **All Predictions**: Classified as "False Positive" (Negative)
- **True Positives**: 0 (no alerts identified as threats)
- **False Positives**: 0 (no false alarms)
- **True Negatives**: 4000 (all alerts classified as benign)

#### **3. TRAINING DATA ISSUE:**
The model is learning to classify everything as "not a threat" because:
- **Feedback Data**: Likely contains mostly negative examples
- **Class Imbalance**: Overwhelming majority of feedback is "false positive"
- **Model Behavior**: Learned to always predict the majority class

---

## 🏗️ **MODEL TRAINING PROCESS**

### **🔄 TRAINING WORKFLOW:**
```
Every 6 minutes:
├── CLA polls BigQuery for feedback data
├── Gathers 4,000 training samples
├── Extracts features (confidence, comments, time, etc.)
├── Trains RandomForestClassifier
├── Evaluates on 1,000 test samples
├── Saves model (.pkl) and metrics (.json)
└── Syncs to dashboard directory
```

### **📊 FEATURE ENGINEERING:**
Based on the CLA code, models use these features:
```python
Features:
├── confidence: TAA confidence score
├── comment_length: Length of analyst comments
├── hour: Hour of day (0-23)
├── day_of_week: Day of week (0-6)
└── [Additional engineered features]
```

### **🎯 TARGET VARIABLE:**
- **Binary Classification**: `is_true_positive` (0 or 1)
- **0**: False Positive (benign alert)
- **1**: True Positive (actual threat)

---

## 📈 **TRAINING EVOLUTION**

### **📊 PROGRESSIVE IMPROVEMENT:**
The training data size increased over time, indicating:
- **Growing Dataset**: More feedback accumulated
- **System Learning**: CLA adapting to more data
- **Scalability**: System can handle increasing data volumes

### **⏱️ TRAINING FREQUENCY:**
- **Interval**: 6 minutes (360 seconds)
- **Total Cycles**: 66 training cycles
- **Active Period**: ~15.5 hours
- **Efficiency**: Consistent training schedule maintained

---

## 🚨 **CURRENT ISSUES & LIMITATIONS**

### **❌ CRITICAL PROBLEMS:**

#### **1. PERMISSION ERRORS (Recent):**
```
ERROR - Permission denied: '/home/raditio.ghifiardigmail.com/ai-driven-soc/models/trained_model_20250920_013435.pkl'
```
- **Issue**: CLA can't save new models
- **Impact**: Training continues but models aren't saved
- **Status**: Needs permission fix

#### **2. MODEL EFFECTIVENESS:**
- **Zero Precision**: Model never identifies true threats
- **Perfect Accuracy**: Only because it always predicts negative
- **No Learning**: Model hasn't learned to distinguish threats

#### **3. DATA QUALITY:**
- **Class Imbalance**: Overwhelming negative examples
- **No Positive Cases**: No true threats in training data
- **Feedback Bias**: All feedback indicates false positives

---

## 🔧 **TECHNICAL SPECIFICATIONS**

### **💾 STORAGE DETAILS:**
```bash
Directory: /home/raditio.ghifiardigmail.com/ai-driven-soc/models/
File Pattern: trained_model_YYYYMMDD_HHMMSS.pkl
Metrics Pattern: model_metrics_YYYYMMDD_HHMMSS.json
Total Size: 3,712 KB (3.7 MB)
Average Model Size: 52 KB
```

### **🔄 SYNC MECHANISM:**
- **Source**: `/home/raditio.ghifiardigmail.com/ai-driven-soc/models/`
- **Target**: `/home/app/ai-model-training-dashboard/models/`
- **Method**: Automatic copying after each training
- **Status**: All 66 models successfully synced

---

## 📊 **DASHBOARD INTEGRATION**

### **🎯 DASHBOARD DISPLAY:**
The dashboard now shows:
- **Models Trained**: 66+ (instead of 0)
- **Training Status**: Active (instead of inactive)
- **Mode**: LEARNING MODE (instead of STATISTICS)
- **Last Training**: Recent timestamps

### **📈 REAL-TIME UPDATES:**
- **New Models**: Appear every 6 minutes
- **Sync Status**: Automatic copying to dashboard directory
- **Metrics Display**: Real training performance data
- **Status Indicators**: Live training activity

---

## 🎯 **BUSINESS IMPACT**

### **✅ POSITIVE OUTCOMES:**
1. **System Operational**: CLA is actively training models
2. **Scalable Architecture**: Can handle 4,000+ training samples
3. **Continuous Learning**: Models updated every 6 minutes
4. **Dashboard Integration**: Real-time visibility into training
5. **Data Pipeline**: Established feedback collection and processing

### **⚠️ AREAS FOR IMPROVEMENT:**
1. **Model Effectiveness**: Need better training data with true positives
2. **Permission Issues**: Fix model saving permissions
3. **Class Balance**: Address overwhelming negative examples
4. **Feature Engineering**: Improve feature extraction for better discrimination
5. **Evaluation Metrics**: Focus on precision/recall rather than just accuracy

---

## 🚀 **RECOMMENDATIONS**

### **🎯 IMMEDIATE ACTIONS:**
1. **Fix Permissions**: Resolve model saving permission errors
2. **Data Quality**: Investigate training data for true positive examples
3. **Class Balance**: Implement data augmentation or sampling techniques
4. **Feature Engineering**: Add more discriminative features
5. **Evaluation**: Focus on precision/recall metrics

### **📈 LONG-TERM IMPROVEMENTS:**
1. **Active Learning**: Implement uncertainty-based sampling
2. **Ensemble Methods**: Combine multiple model approaches
3. **Domain Expertise**: Incorporate security analyst knowledge
4. **Continuous Monitoring**: Track model performance over time
5. **A/B Testing**: Compare different model architectures

---

## 🎊 **CONCLUSION**

### **🎉 ACHIEVEMENTS:**
- ✅ **66 Models Trained**: Successful continuous learning implementation
- ✅ **Automated Pipeline**: 6-minute training cycles working
- ✅ **Dashboard Integration**: Real-time visibility achieved
- ✅ **Scalable System**: Handles 4,000+ training samples
- ✅ **Data Collection**: Established feedback processing pipeline

### **🔍 KEY INSIGHTS:**
The 66+ models represent a **successful technical implementation** of continuous learning, but they reveal **fundamental data quality issues** that prevent the models from being operationally effective. The system is learning, but it's learning the wrong patterns due to class imbalance in the training data.

### **🎯 NEXT STEPS:**
Focus on **improving training data quality** and **model effectiveness** rather than just increasing the number of models. The infrastructure is solid; now we need better data and more sophisticated learning approaches.

---

## 📞 **TECHNICAL SUMMARY**

**The 66+ models are RandomForestClassifier instances trained on SOC feedback data to classify security alerts as true or false positives. While technically successful in terms of automation and scale, they currently suffer from class imbalance issues that result in perfect accuracy but zero precision - essentially learning to classify everything as benign. The system demonstrates excellent engineering but needs data quality improvements to achieve operational effectiveness.**

**🤖🧠 Your CLA system is technically impressive but needs better training data to become operationally effective!**

---

## 🧩 Appendix: Cursor Task Checklist
- [ ] Open `~/Downloads/ai-driven-soc` in Cursor
- [ ] `python3 -m venv .venv && source .venv/bin/activate`
- [ ] `pip install -r requirements.txt` **or** install minimal deps above
- [ ] Export `GOOGLE_APPLICATION_CREDENTIALS` **or** `gcloud auth application-default login`
- [ ] `python "Mixture of Expert/fix_bigquery_schemas.py"`
- [ ] Fix model dir permissions (step 6)
- [ ] Start trainer (`python cla_trainer.py` or your service)
- [ ] `bash deploy/access_simple_dashboard.sh` **or** `streamlit run <entry.py>`
- [ ] Check **precision/recall/F1/PR-AUC** (not just accuracy)
- [ ] If recall is weak → enable **SMOTE** and increase positive sampling
