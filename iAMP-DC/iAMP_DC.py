import numpy as np  
import pandas as pd  
from sklearn.model_selection import KFold  
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, roc_curve  
import tensorflow as tf  
from tensorflow.keras.models import Model  
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Conv1D, Bidirectional, LSTM, Concatenate, Layer, GlobalAveragePooling1D, Reshape, multiply, RepeatVector, Permute, Lambda, GlobalMaxPooling1D  
import tensorflow.keras.backend as K  
from tensorflow.keras.optimizers import Adam  
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping  
import matplotlib.pyplot as plt  

# ---------------------- 1. 数据加载 ----------------------  
def load_data(csv_path):  
    data = pd.read_csv(csv_path)  
    sequences = data['sequence'].values  
    labels = data['label'].values  
    return sequences, labels  

# ---------------------- 2. 特征提取 ----------------------  
def extract_aac(sequences):  
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"  
    features = []  
    for seq in sequences:  
        feat = [seq.count(aa) for aa in amino_acids]  
        total = sum(feat)  
        feat = np.array(feat) / total if total > 0 else np.zeros(len(feat))  
        features.append(feat)  
    return np.array(features)  

def encode_sequences(sequences, maxlen=100, amino_acid_to_int=None):  
    encoded = []  
    for seq in sequences:  
        enc = [amino_acid_to_int.get(aa, 0) for aa in seq]  
        enc = enc[:maxlen]  
        if len(enc) < maxlen:  
            enc += [0] * (maxlen - len(enc))  
        encoded.append(enc)  
    return np.array(encoded)  

# ---------------------- 3. 定义SE Attention层 ----------------------  
class SEAttention(Layer):  
    def __init__(self, channels, reduction=16):  
        super(SEAttention, self).__init__()  
        self.global_avg_pool = GlobalAveragePooling1D()  
        self.fc1 = Dense(channels // reduction, activation='relu')  
        self.fc2 = Dense(channels, activation='sigmoid')  
        self.reshape = Reshape((1, channels))  
    def call(self, inputs):  
        se = self.global_avg_pool(inputs)  
        se = self.fc1(se)  
        se = self.fc2(se)  
        se = self.reshape(se)  
        return inputs * se  

# ---------------------- 4. 构建模型 ----------------------  
def build_cnn_bilstm_se(input_dim_aac, input_dim_seq, vocab_size, embedding_dim=50):
    # 输入
    aac_input = Input(shape=(input_dim_aac,))
    seq_input = Input(shape=(input_dim_seq,))

    # Embedding层
    embed = Embedding(vocab_size, embedding_dim, input_length=input_dim_seq)(seq_input)

    # CNN层
    conv = Conv1D(64, 3, activation='relu', padding='same')(embed)

    # BiLSTM
    biLSTM = Bidirectional(LSTM(512, return_sequences=True))(conv)
    biLSTM = Bidirectional(LSTM(256, return_sequences=True))(conv)
    biLSTM = Bidirectional(LSTM(128, return_sequences=True))(conv)
    biLSTM = Bidirectional(LSTM(64, return_sequences=True))(conv)

    # 添加SE注意力
    se_layer = SEAttention(channels=biLSTM.shape[-1], reduction=16)
    attn = se_layer(biLSTM)

    # 重定义重复 AAC 特征
    def repeat_aac(x):
        aac, attn_tensor = x
        time_steps = K.shape(attn_tensor)[1]
        aac = K.expand_dims(aac, axis=1)
        aac_repeated = K.tile(aac, [1, time_steps, 1])
        return aac_repeated

    aac_expanded = Lambda(repeat_aac)([aac_input, attn])  # (batch, time_steps, aac_dim)

    # 拼接
    merged = Concatenate()([aac_expanded, attn])
    
    # 新增全局池化层解决形状问题
    x = Dense(512, activation='relu')(merged)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = GlobalAveragePooling1D()(x)  # 全局平均池化
    
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[aac_input, seq_input], outputs=output)
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ---------------------- 5. 交叉验证 ----------------------  
def cross_validate(aac_feats, seq_feats, labels, n_splits=10):  
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)  
    results = []  
    roc_data = []  

    for fold, (train_idx, val_idx) in enumerate(kf.split(aac_feats)):  
        print(f'Fold {fold + 1}')  
        X_aac_train, X_aac_val = aac_feats[train_idx], aac_feats[val_idx]  
        X_seq_train, X_seq_val = seq_feats[train_idx], seq_feats[val_idx]  
        y_train, y_val = labels[train_idx], labels[val_idx]  

        model = build_cnn_bilstm_se(aac_feats.shape[1], seq_feats.shape[1], vocab_size=21)  

        # 回调  
        reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)  
        earlyStop = EarlyStopping(monitor='val_loss', patience=5)  

        # 训练  
        history = model.fit([X_aac_train, X_seq_train], y_train,  
                  epochs=50, batch_size=32,  
                  validation_data=([X_aac_val, X_seq_val], y_val),  
                  callbacks=[reduceLR, earlyStop],  
                  verbose=2)  

        # 预测  
        y_pred = model.predict([X_aac_val, X_seq_val]).flatten()  # 展平输出
        y_pred_class = (y_pred > 0.5).astype(int)  

        # 评价指标  
        acc = accuracy_score(y_val, y_pred_class)  
        sens = recall_score(y_val, y_pred_class)  
        spec = recall_score(y_val, y_pred_class, pos_label=0)  
        mcc = matthews_corrcoef(y_val, y_pred_class)  
        f1 = f1_score(y_val, y_pred_class)  
        roc_auc = roc_auc_score(y_val, y_pred)  

        results.append([fold+1, acc, sens, spec, mcc, f1, roc_auc])  
        # ROC  
        fpr, tpr, _ = roc_curve(y_val, y_pred)  
        roc_data.append((fpr, tpr, roc_auc))  
    return results, roc_data  

# ---------------------- 6. 结果保存 & ROC绘图 ----------------------  
def save_results(results, filename):  
    df = pd.DataFrame(results, columns=['Fold', 'Accuracy', 'Sensitivity', 'Specificity', 'MCC', 'F1-Score', 'ROC_AUC'])  
    df.loc['Average'] = df.mean()  
    df.to_csv(filename, index=False)  

def plot_roc(roc_data):  
    plt.figure()  
    for i, (fpr, tpr, auc) in enumerate(roc_data):  
        plt.plot(fpr, tpr, lw=2, label=f'Fold {i+1} (AUC={auc:.2f})')  
    plt.plot([0,1], [0,1], 'k--')  
    plt.xlabel('False Positive Rate')  
    plt.ylabel('True Positive Rate')  
    plt.title('ROC Curve')  
    plt.legend(loc='lower right')  
    plt.show()  

# ---------------------- 7. 主流程 ----------------------  
def main():  
    # 数据路径  
    csv_path = 'combined_data_.csv'  
    output_file = 'iAMP_DC.csv'  

    sequences, labels = load_data(csv_path)  

    # AAC特征  
    aac_feats = extract_aac(sequences)  

    # 字符编码  
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"  
    amino_acid_to_int = {aa: i+1 for i, aa in enumerate(amino_acids)}  # 0为填充  
    maxlen = 100  
    seq_feats = encode_sequences(sequences, maxlen, amino_acid_to_int)  

    print("AAC 特征形状：", aac_feats.shape)
    print("序列编码形状：", seq_feats.shape)
    print("标签形状：", labels.shape)

    # 交叉验证
    results, roc_data = cross_validate(aac_feats, seq_feats, labels, n_splits=10)

    # 保存结果
    save_results(results, output_file)

    # ROC曲线
    plot_roc(roc_data)

if __name__ == "__main__":
    main()