import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

# Carregar o conjunto de dados MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Normalizar os dados
x_train, x_val, x_test = x_train / 255.0, x_val / 255.0, x_test / 255.0

# Parâmetros do ACO
num_ants = 10  # Número de formigas
num_iterations = 20  # Número de iterações
decay = 0.95  # Taxa de decaimento do feromônio
alpha = 1  # Peso do feromônio
beta = 2  # Peso da heurística

# Limites dos hiperparâmetros
bounds = np.array([
    [0.0001, 0.1],  # Taxa de aprendizado
    [32, 256],      # Tamanho do batch
    [16, 256]       # Número de neurônios
])

# Inicialização do feromônio
pheromone = np.ones((num_ants, len(bounds))) / num_ants

# Inicialização das posições das formigas
positions = np.random.uniform(bounds[:, 0], bounds[:, 1], (num_ants, len(bounds)))

# Melhores posições e pontuações
personal_best_positions = np.copy(positions)
personal_best_scores = np.full(num_ants, -np.inf)
global_best_position = None
global_best_score = -np.inf

# Função para avaliar o desempenho do MLP
def evaluate_mlp(params):
    learning_rate, batch_size, neurons = params
    batch_size, neurons = int(batch_size), int(neurons)

    # Construir o modelo MLP
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(neurons, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compilar o modelo
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Treinar o modelo
    model.fit(x_train, y_train, epochs=5, batch_size=batch_size, verbose=0, validation_data=(x_val, y_val))

    # Avaliar o modelo
    _, accuracy = model.evaluate(x_val, y_val, verbose=0)
    return accuracy

# Loop do ACO
for iteration in range(num_iterations):
    for i in range(num_ants):
        # Avaliar a posição atual da formiga
        score = evaluate_mlp(positions[i])

        # Atualizar a melhor posição pessoal
        if score > personal_best_scores[i]:
            personal_best_scores[i] = score
            personal_best_positions[i] = positions[i]

            # Atualizar a melhor posição global
            if score > global_best_score:
                global_best_score = score
                global_best_position = positions[i]

        # Atualizar o feromônio
        pheromone[i] = (1 - decay) * pheromone[i] + decay * score

    # Movimentar as formigas
    for i in range(num_ants):
        for j in range(len(bounds)):
            # Escolher a próxima posição com base no feromônio e na heurística
            probabilities = pheromone[:, j] ** alpha * (1 / (positions[:, j] + 1e-10)) ** beta
            probabilities /= probabilities.sum()
            positions[i, j] = np.random.choice(positions[:, j], p=probabilities)

    # Garantir que as posições estejam dentro dos limites
    positions = np.clip(positions, bounds[:, 0], bounds[:, 1])

    print(f"Iteração {iteration}: Melhor Acurácia = {global_best_score:.6f}")

# Resultado final
print(f"Melhores Hiperparâmetros Encontrados:")
print(f"Taxa de Aprendizado = {global_best_position[0]:.6f}")
print(f"Tamanho do Batch = {int(global_best_position[1])}")
print(f"Número de Neurônios = {int(global_best_position[2])}")