import os

from sklearn import svm
import cv2
import numpy as np
from tqdm import tqdm
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from skimage.feature import hog, local_binary_pattern



def extract_features(image_path,win_size=(256,256)):
    """从图像路径提取 HOG 特征"""

    img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    hog = cv2.HOGDescriptor( 
        win_size,
        (16,16),
        (8,8),
        (16,16),
        9
    )
    hog_feat = hog.compute(img).flatten().astype(np.float32)

    lbp_matrix = local_binary_pattern(img, P=8, R=1, method='uniform')
    n_bins = 8 + 2
    lbp_hist, _ = np.histogram(lbp_matrix.ravel(), bins=n_bins, range=(0, n_bins))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= (lbp_hist.sum() + 1e-7)
    
    hsv = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    color_hist = np.hstack([h_hist, s_hist, v_hist])
    color_hist = color_hist.astype(np.float32)
    color_hist /= (color_hist.sum() + 1e-7)
    combined = np.hstack([hog_feat, lbp_hist, color_hist])
    return combined

def load_data(folder,label):
    feature=[]
   
    files = os.listdir(folder)
    for file in tqdm(files) :
        path = os.path.join(folder,file)
        features = extract_features(path)
        if features is not None:
            feature.append(features)
            
    x = np.array(feature,dtype = np.float32)
    y = np.full(len(x), label, dtype=np.int8)
    return x,y

script_dir = os.path.dirname(os.path.abspath(__file__))
test_real = os.path.join(os.path.dirname(script_dir), "deep_face", "test", "real")
test_fake = os.path.join(os.path.dirname(script_dir), "deep_face", "test", "fake")

x_test_real, y_test_real = load_data(test_real,1)
x_test_fake, y_test_fake = load_data(test_fake,0)

X_test = np.vstack([x_test_real, x_test_fake])
y_test = np.hstack([y_test_real, y_test_fake])

print(X_test.shape)
print(y_test.shape)

MODEL_DIR="models"
MODEL_PATH=os.path.join(MODEL_DIR,"ml_model.joblib")
SCALER_PATH=os.path.join(MODEL_DIR,"scaler.joblib")
PCA_PATH=os.path.join(MODEL_DIR,"pca.joblib")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
pca = joblib.load(PCA_PATH)

X_test_scaled = scaler.transform(X_test)
X_pca = pca.transform(X_test_scaled)
y_pred = model.predict(X_pca)

acc = accuracy_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

print(acc)
print(f1)