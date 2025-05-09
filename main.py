import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Загрузка данных
df = pd.read_csv("sample_data/Global_Cybersecurity_Threats_2015-2024.csv")

# Удаляем пропущенные значения
df = df.dropna()

# Названия колонок (проверь в df.columns, если отличаются)
target_col = 'Target Industry'

# Автоматически определим строковые колонки и закодируем
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Кодируем целевую переменную (если ещё не закодирована)
target_encoder = LabelEncoder()
df[target_col] = target_encoder.fit_transform(df[target_col])

# Разделяем данные
X = df.drop(target_col, axis=1)
y = df[target_col]

# Масштабируем признаки
X = StandardScaler().fit_transform(X)

# Делим выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Преобразуем данные в тензоры
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)
y_test_tensor  = torch.tensor(y_test.to_numpy(), dtype=torch.long)

# Архитектура сети
class CyberThreatNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CyberThreatNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = CyberThreatNN(input_size=X.shape[1], num_classes=len(set(y)))

# Обучение
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 50

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Тестирование
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_classes = predictions.argmax(dim=1)
    print("\nКлассификационный отчёт:")
    print(classification_report(y_test_tensor, predicted_classes))
