import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 数据预处理类
class PowerDataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def load_and_process_data(self, train_path, test_path):
        """加载并处理数据"""
        try:
            # 读取数据
            print(f"Loading data from {train_path} and {test_path}")
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            print(f"Train data shape: {train_data.shape}")
            print(f"Test data shape: {test_data.shape}")
            
            # 合并数据以便统一处理
            all_data = pd.concat([train_data, test_data], ignore_index=True)
            
            # 数据处理
            processed_data = self.process_features(all_data)
            
            # 分割回训练和测试数据
            train_size = len(train_data)
            train_processed = processed_data[:train_size]
            test_processed = processed_data[train_size:]
            
            print(f"\nProcessed train data shape: {train_processed.shape}")
            print(f"Processed test data shape: {test_processed.shape}")
            
            return train_processed, test_processed
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Please check if the data files exist and have the correct format")
            raise
    
    def process_features(self, data):
        """特征处理"""
        # 复制数据
        processed = data.copy()
        
        # 定义数值列
        numeric_columns = [
            'global_active_power', 'global_reactive_power', 'voltage', 
            'global_intensity', 'sub_metering_1', 'sub_metering_2', 
            'sub_metering_3', 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
        ]
        
        # 转换数值列类型并处理缺失值
        for col in numeric_columns:
            if col in processed.columns:
                # 将非数值字符串转换为NaN
                processed[col] = pd.to_numeric(processed[col], errors='coerce')
                # 用列的中位数填充缺失值
                if processed[col].isna().any():
                    median_value = processed[col].median()
                    processed[col].fillna(median_value, inplace=True)
                    print(f"Warning: {col} contains missing values, filled with median: {median_value}")
        
        # 计算sub_metering_remainder
        processed['sub_metering_remainder'] = (
            processed['global_active_power'] * 1000 / 60 - 
            processed['sub_metering_1'] - 
            processed['sub_metering_2'] - 
            processed['sub_metering_3']
        )
        
        # 处理降水数据（除以10）
        if 'RR' in processed.columns:
            processed['RR'] = processed['RR'] / 10
        
        # 添加时间特征
        if 'date' in processed.columns:
            try:
                processed['date'] = pd.to_datetime(processed['date'])
                processed['month'] = processed['date'].dt.month
                
                # 周期性编码
                processed['month_sin'] = np.sin(2 * np.pi * processed['month'] / 12)
                processed['month_cos'] = np.cos(2 * np.pi * processed['month'] / 12)
                
                processed['day_of_year'] = processed['date'].dt.dayofyear
                processed['day_sin'] = np.sin(2 * np.pi * processed['day_of_year'] / 365)
                processed['day_cos'] = np.cos(2 * np.pi * processed['day_of_year'] / 365)
                
                processed['weekday'] = processed['date'].dt.weekday
                processed['weekday_sin'] = np.sin(2 * np.pi * processed['weekday'] / 7)
                processed['weekday_cos'] = np.cos(2 * np.pi * processed['weekday'] / 7)
            except Exception as e:
                print(f"Warning: Could not parse date column: {e}")
        
        # 检查是否还有非数值数据
        for col in numeric_columns + ['sub_metering_remainder']:
            if col in processed.columns:
                if not pd.api.types.is_numeric_dtype(processed[col]):
                    print(f"Warning: {col} is still not numeric after conversion")
        
        return processed
    
    def create_sequences(self, data, input_length, output_length, target_column='global_active_power'):
        """创建时间序列数据：输入序列 + 输出序列"""
        # 选择特征列
        feature_columns = [
            'global_active_power', 'global_reactive_power', 'voltage', 
            'global_intensity', 'sub_metering_1', 'sub_metering_2', 
            'sub_metering_3', 'sub_metering_remainder', 'RR', 
            'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
        ]
        
        # 添加时间特征（如果存在）
        time_features = ['month_sin', 'month_cos', 'day_sin', 'day_cos', 'weekday_sin', 'weekday_cos']
        for tf in time_features:
            if tf in data.columns:
                feature_columns.append(tf)
        
        # 确保所有特征列都存在且为数值类型
        available_features = []
        for col in feature_columns:
            if col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    available_features.append(col)
                else:
                    print(f"Warning: Skipping non-numeric column {col}")
        
        print(f"Using features: {available_features}")
        
        # 检查是否有足够的特征
        if len(available_features) == 0:
            raise ValueError("No valid numeric features found!")
        
        features = data[available_features].values
        target = data[target_column].values
        
        # 检查数据中是否还有NaN值
        if np.isnan(features).any():
            print("Warning: NaN values found in features, replacing with 0")
            features = np.nan_to_num(features, nan=0.0)
        
        if np.isnan(target).any():
            print("Warning: NaN values found in target, replacing with median")
            target_median = np.nanmedian(target)
            target = np.nan_to_num(target, nan=target_median)
        
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(len(features_scaled) - input_length - output_length + 1):
            X.append(features_scaled[i:i + input_length])
            y.append(target[i + input_length:i + input_length + output_length])
        
        return np.array(X), np.array(y)

# 数据集类
class PowerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# LSTM模型 - 多步预测
class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # 取最后一个时间步的输出
        out = self.fc(out)
        return out

# Transformer模型 - 多步预测
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # 全局平均池化
        x = self.dropout(x)
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 改进的GRU-Transformer混合模型 - 多步预测
class GRUTransformerModel(nn.Module):
    def __init__(self, input_size, output_size, d_model=128, nhead=8, num_layers=2, 
                 gru_hidden_size=64, gru_layers=1, dropout=0.2):
        super(GRUTransformerModel, self).__init__()
        
        # GRU层用于提取序列特征
        self.gru = nn.GRU(
            input_size, 
            gru_hidden_size, 
            num_layers=gru_layers, 
            batch_first=True, 
            dropout=dropout if gru_layers > 1 else 0
        )
        self.gru_norm = nn.LayerNorm(gru_hidden_size)
        
        # 投影到Transformer维度
        self.gru_projection = nn.Linear(gru_hidden_size, d_model)
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Transformer编码器
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 特征融合和输出
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_size)
        )
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.dim() < 2:  # 跳过一维参数（如偏置）
                continue
                
            if 'gru' in name and 'weight' in name:
                # GRU权重使用正交初始化
                if 'weight_hh' in name:  # 隐藏层到隐藏层的权重
                    nn.init.orthogonal_(param)
                else:  # 输入到隐藏层的权重
                    nn.init.xavier_uniform_(param)
            elif 'weight' in name:
                # 其他权重使用Xavier初始化
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # 偏置项初始化为零
                nn.init.constant_(param, 0.0)
    
    def forward(self, x):
        batch_size, seq_len, input_size = x.size()
        
        # GRU特征提取
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, gru_hidden_size)
        gru_out = self.gru_norm(gru_out)
        gru_features = self.gru_projection(gru_out)  # (batch_size, seq_len, d_model)
        
        # 原始特征投影
        x_proj = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # 位置编码
        x_proj = self.pos_encoder(x_proj)
        gru_features = self.pos_encoder(gru_features)
        
        # Transformer编码
        transformer_out = self.transformer(x_proj)
        gru_transformer_out = self.transformer(gru_features)
        
        # 特征融合 - 取序列平均
        transformer_pool = transformer_out.mean(dim=1)  # (batch_size, d_model)
        gru_pool = gru_transformer_out.mean(dim=1)      # (batch_size, d_model)
        
        # 拼接特征
        combined = torch.cat([transformer_pool, gru_pool], dim=1)  # (batch_size, d_model * 2)
        
        # 输出
        out = self.fusion(combined)
        return out

# 训练函数
def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, device='cpu', model_name='model'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 创建保存模型的目录
    os.makedirs('saved_models', exist_ok=True)
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # 打印训练进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'saved_models/best_{model_name}.pth')
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(f'saved_models/best_{model_name}.pth'))
    return model, train_losses, val_losses

# 评估函数 - 多步预测评估
def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            
            # 保存整个序列的预测结果
            predictions.append(outputs.cpu().numpy())
            actuals.append(batch_y.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    # 计算整体MSE和MAE
    mse = mean_squared_error(actuals.flatten(), predictions.flatten())
    mae = mean_absolute_error(actuals.flatten(), predictions.flatten())
    
    return mse, mae, predictions, actuals

# 可视化预测结果
def plot_predictions(actuals, predictions, title, num_samples=3):
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        idx = np.random.randint(0, len(actuals))
        plt.subplot(num_samples, 1, i+1)
        plt.plot(actuals[idx], label='Actual', linewidth=2)
        plt.plot(predictions[idx], label='Predicted', alpha=0.8)
        plt.title(f'Sample {i+1}')
        plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.close()

# 主实验函数
def run_experiments(train_path, test_path, input_length=90, output_length=90, 
                   num_experiments=5, prediction_type="short_term"):
    """运行完整实验"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据处理
    processor = PowerDataProcessor()
    train_data, test_data = processor.load_and_process_data(train_path, test_path)
    
    # 创建序列数据
    X_train, y_train = processor.create_sequences(train_data, input_length, output_length)
    X_test, y_test = processor.create_sequences(test_data, input_length, output_length)
    
    # 划分训练集和验证集
    split_idx = int(len(X_train) * 0.8)
    X_train_split, X_val = X_train[:split_idx], X_train[split_idx:]
    y_train_split, y_val = y_train[:split_idx], y_train[split_idx:]
    
    input_size = X_train.shape[2]
    
    # 存储结果
    results = {
        'LSTM': {'mse': [], 'mae': []},
        'Transformer': {'mse': [], 'mae': []},
        'GRU-Transformer': {'mse': [], 'mae': []}  # 修改模型名称
    }
    
    print(f"\n开始{output_length}天预测实验 - 输入序列长度: {input_length}, 输出序列长度: {output_length}")
    print(f"训练数据形状: {X_train.shape}, 测试数据形状: {X_test.shape}")
    
    for exp in range(num_experiments):
        print(f"\n=== 实验 {exp+1}/{num_experiments} ===")
        set_seed(42 + exp)
        
        # 创建数据加载器
        train_dataset = PowerDataset(X_train_split, y_train_split)
        val_dataset = PowerDataset(X_val, y_val)
        test_dataset = PowerDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # 1. LSTM模型
        print("\n训练LSTM模型...")
        lstm_model = LSTMModel(input_size, output_size=output_length,
                              hidden_size=256 if output_length > 100 else 128).to(device)
        lstm_model, _, _ = train_model(
            lstm_model, train_loader, val_loader,
            num_epochs=150 if output_length > 100 else 100,
            device=device,
            model_name=f"lstm_{prediction_type}_exp{exp+1}"
        )
        mse, mae, preds, actuals = evaluate_model(lstm_model, test_loader, device)
        results['LSTM']['mse'].append(mse)
        results['LSTM']['mae'].append(mae)
        print(f"LSTM - MSE: {mse:.6f}, MAE: {mae:.6f}")
        
        # 可视化第一轮的预测结果
        if exp == 0:
            plot_predictions(actuals, preds, f"LSTM {output_length}天预测结果")
        
        # 2. Transformer模型
        print("\n训练Transformer模型...")
        transformer_model = TransformerModel(input_size, output_size=output_length,
                                           d_model=256 if output_length > 100 else 128).to(device)
        transformer_model, _, _ = train_model(
            transformer_model, train_loader, val_loader,
            num_epochs=150 if output_length > 100 else 100,
            device=device,
            model_name=f"transformer_{prediction_type}_exp{exp+1}"
        )
        mse, mae, preds, actuals = evaluate_model(transformer_model, test_loader, device)
        results['Transformer']['mse'].append(mse)
        results['Transformer']['mae'].append(mae)
        print(f"Transformer - MSE: {mse:.6f}, MAE: {mae:.6f}")
        
        # 可视化第一轮的预测结果
        if exp == 0:
            plot_predictions(actuals, preds, f"Transformer {output_length}天预测结果")
        
        # 3. GRU-Transformer混合模型（替换CNN-Transformer）
        print("\n训练GRU-Transformer模型...")
        gru_transformer_model = GRUTransformerModel(
            input_size, 
            output_size=output_length,
            d_model=256 if output_length > 100 else 128,
            gru_hidden_size=128 if output_length > 100 else 64
        ).to(device)
        
        gru_transformer_model, _, _ = train_model(
            gru_transformer_model, train_loader, val_loader, 
            num_epochs=150 if output_length > 100 else 100,
            device=device, 
            model_name=f"gru_transformer_{prediction_type}_exp{exp+1}"
        )
        mse, mae, preds, actuals = evaluate_model(gru_transformer_model, test_loader, device)
        results['GRU-Transformer']['mse'].append(mse)
        results['GRU-Transformer']['mae'].append(mae)
        print(f"GRU-Transformer - MSE: {mse:.6f}, MAE: {mae:.6f}")
        
        # 可视化第一轮的预测结果
        if exp == 0:
            plot_predictions(actuals, preds, f"GRU-Transformer {output_length}天预测结果")
    
    # 计算平均值和标准差
    print(f"\n=== {output_length}天预测结果汇总 ===")
    for model_name, metrics in results.items():
        mse_mean = np.mean(metrics['mse'])
        mse_std = np.std(metrics['mse'])
        mae_mean = np.mean(metrics['mae'])
        mae_std = np.std(metrics['mae'])
        
        print(f"{model_name}:")
        print(f"  MSE: {mse_mean:.6f} ± {mse_std:.6f}")
        print(f"  MAE: {mae_mean:.6f} ± {mae_std:.6f}")
    
    return results

if __name__ == "__main__":
    # 数据文件路径
    train_path = "train.csv"
    test_path = "test.csv"
    
    print("电力消耗预测任务开始...")
    
    # 短期预测（90天）
    print("\n" + "="*50)
    print("短期预测（未来90天）")
    print("="*50)
    short_term_results = run_experiments(
        train_path, test_path, 
        input_length=90, output_length=90,
        num_experiments=5, prediction_type="short_term"
    )
    
    # 长期预测（365天）
    print("\n" + "="*50)
    print("长期预测（未来365天）")
    print("="*50)
    long_term_results = run_experiments(
        train_path, test_path, 
        input_length=90, output_length=365,
        num_experiments=5, prediction_type="long_term"
    )
    
    print("\n实验完成！")
    print("短期预测结果:")
    for model, metrics in short_term_results.items():
        print(f"{model}: MSE={np.mean(metrics['mse']):.6f} ± {np.std(metrics['mse']):.6f}, "
              f"MAE={np.mean(metrics['mae']):.6f} ± {np.std(metrics['mae']):.6f}")
    
    print("\n长期预测结果:")
    for model, metrics in long_term_results.items():
        print(f"{model}: MSE={np.mean(metrics['mse']):.6f} ± {np.std(metrics['mse']):.6f}, "
              f"MAE={np.mean(metrics['mae']):.6f} ± {np.std(metrics['mae']):.6f}")