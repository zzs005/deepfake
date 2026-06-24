import os

from sklearn import svm
import cv2
import numpy as np
from tqdm import tqdm
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC

from skimage.feature import hog, local_binary_pattern




def extract_hog(image_path,win_size=(256,256)):
    """从图像路径提取 HOG 特征"""

    img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    hog = cv2.HOGDescriptor( 
        win_size,
        (16,16),
        (8,8),
        (8,8),
        9
    )
    hog_feat = hog.compute(img).flatten().astype(np.float32)

    lbp_matrix = local_binary_pattern(img, P=8, R=1, method='uniform')
    n_bins = 8 + 2
    lbp_hist, _ = np.histogram(lbp_matrix.ravel(), bins=n_bins, range=(0, n_bins))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= (lbp_hist.sum() + 1e-7)
    
    combined = np.hstack([hog_feat, lbp_hist])
    return combined


def load_data(folder,label):
    feature=[]
   
    files = os.listdir(folder)
    for file in tqdm(files) :
        path = os.path.join(folder,file)
        features = extract_hog(path)
        if features is not None:
            feature.append(features)
            
    x = np.array(feature,dtype = np.float32)
    y = np.full(len(x), label, dtype=np.int8)
    return x,y

script_dir = os.path.dirname(os.path.abspath(__file__))
train_real = os.path.join(os.path.dirname(script_dir), "deep_face", "train", "real")
train_fake = os.path.join(os.path.dirname(script_dir), "deep_face", "train", "fake")

x_train_real, y_train_real = load_data(train_real,1)
x_train_fake, y_train_fake = load_data(train_fake,0)

X_train = np.vstack([x_train_real, x_train_fake])
y_train = np.hstack([y_train_real, y_train_fake])

print(X_train.shape)
print(y_train.shape)
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
pca = PCA(n_components=5000)
X_pca = pca.fit_transform(X_train_scaler)
model = LinearSVC(C=1)
model.fit(X_pca,y_train)

MODEL_DIR="models"
MODEL_PATH=os.path.join(MODEL_DIR,"ml_model.joblib")
SCALER_PATH=os.path.join(MODEL_DIR,"scaler.joblib")
PCA_PATH=os.path.join(MODEL_DIR,"pca.joblib")
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(pca, PCA_PATH)


