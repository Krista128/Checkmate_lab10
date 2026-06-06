import chess.engine
import pandas as pd
import numpy as np
from enginx import evaluate_position
from FENtoTensor import fen_to_tensor, result_to_label, extract_fen_pairs
from model import create_error_classifier_model
import ast
from sklearn.model_selection import train_test_split
from like_sunfish import evaluate_sunfish_style
import gc



print('===Чтение файлов...===')
p1 = pd.read_csv("part1.csv")
p2 = pd.read_csv("part2.csv")
p3 = pd.read_csv("part3.csv")

print('===Объединение датафреймов...===')
#data = pd.concat([p1, p2, p3], ignore_index=True)
data = p1.copy()
print(f'Всего партий: {len(data)}')

data['fen_list'] = data['positions'].apply(ast.literal_eval)

print('===Создание пар позиций (FEN, Next_FEN)...===')

all_pairs = []
all_labels = []

for idx, row in data.iterrows():
    fen_list = row['fen_list']
    result = row['result']
    
    label = result_to_label(result)
    pairs = extract_fen_pairs(fen_list)
    
    for current_fen, next_fen in pairs:
        all_pairs.append((current_fen, next_fen))
        all_labels.append(label)
    
    if idx % 1000 == 0:
        print(f"Обработано {idx} партий...")

print(f'Всего пар позиций: {len(all_pairs)}')


def is_mistake(current_fen, next_fen, threshold=-100):
    score_before = evaluate_sunfish_style(current_fen)
    score_after = evaluate_sunfish_style(next_fen)
    return (score_after - score_before) < threshold


print('===Вычисление ошибок для каждой пары...===')

error_labels = []
for i, (current_fen, next_fen) in enumerate(all_pairs):
    is_err = is_mistake(current_fen, next_fen, -100)
    error_labels.append(is_err)
    
    if (i + 1) % 1000 == 0:
        print(f"Обработано {i+1}/{len(all_pairs)} пар...")



error_labels = np.array(error_labels).reshape(-1, 1)
print(f'Ошибок: {error_labels.sum()} / {len(error_labels)} ({error_labels.mean()*100:.1f}%)')


print('===Разделение пар на выборки (без преобразования)...===')

indices = np.arange(len(all_pairs))
train_idx, temp_idx = train_test_split(
    indices, test_size=0.3, random_state=42, stratify=error_labels
)

val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, random_state=42, stratify=error_labels[temp_idx]
)

train_labels = error_labels[train_idx].flatten()
val_labels = error_labels[val_idx].flatten()
test_labels = error_labels[test_idx].flatten()

print(f'Train: {len(train_idx)} пар ({len(train_idx)/len(indices)*100:.1f}%)')
print(f'Val: {len(val_idx)} пар ({len(val_idx)/len(indices)*100:.1f}%)')
print(f'Test: {len(test_idx)} пар ({len(test_idx)/len(indices)*100:.1f}%)')
print(f'Ошибок в train: {train_labels.sum()} / {len(train_labels)} ({train_labels.mean()*100:.1f}%)')

del error_labels, indices
gc.collect()

print('\n===Создание генератора батчей...===')

def data_generator(all_pairs, pair_indices, labels, batch_size=32, shuffle=True):
    num_samples = len(pair_indices)
    
    while True:  
        if shuffle:
            
            shuffled_perm = np.random.permutation(num_samples)
            shuffled_pair_indices = pair_indices[shuffled_perm]
            shuffled_labels = labels[shuffled_perm]
        else:
            shuffled_pair_indices = pair_indices
            shuffled_labels = labels
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            
            current_batch_indices = shuffled_pair_indices[start_idx:end_idx]
            current_batch_labels = shuffled_labels[start_idx:end_idx]
            
            # Преобразуем только этот батч в тензоры (8, 8, 12)
            batch_X = np.array(
                [fen_to_tensor(all_pairs[i][0]) for i in current_batch_indices],
                dtype=np.float32
            )
            batch_y = current_batch_labels.astype(np.float32).reshape(-1, 1)
            
            yield batch_X, batch_y


print('===Создание модели...===')
model = create_error_classifier_model(input_shape=(8, 8, 12))
model.summary()

print('===Обучение модели...===')
BATCH_SIZE = 64
EPOCHS = 10

train_generator = data_generator(
    all_pairs, train_idx, train_labels,
    batch_size=BATCH_SIZE, shuffle=True
)

val_generator = data_generator(
    all_pairs, val_idx, val_labels,
    batch_size=BATCH_SIZE, shuffle=False
)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=len(train_idx) // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=len(val_idx) // BATCH_SIZE,
    verbose=1
)

model.save('chess_error_model.h5')
print('Модель сохранена в chess_error_model.h5')
