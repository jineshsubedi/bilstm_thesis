from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
import tensorflow as tf
import keras_tuner as kt

def build_bilstm_model_fixed(hp):
    model = Sequential()
    
    # Reduced max_value from 512 to 256 to prevent overfitting and speed up search
    # given the input shape (10, 44)
    model.add(Bidirectional(LSTM(
        hp.Int('lstm_units1', min_value=128, max_value=256, step=32),
        return_sequences=True,
        recurrent_dropout=hp.Float('recurrent_dropout1', 0.0, 0.5, step=0.1),
        activation='tanh'
    ), input_shape=(10, 44))) # Assuming X_train_seq_final.shape[1:] is (10, 44)
    
    model.add(LayerNormalization())
    model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))
    
    # Reduced max_value from 256 to 128
    model.add(Bidirectional(LSTM(
        hp.Int('lstm_units2', min_value=64, max_value=128, step=32),
        recurrent_dropout=hp.Float('recurrent_dropout2', 0.1, 0.5, step=0.1),
        activation='tanh'
    )))
    model.add(LayerNormalization())
    model.add(Dropout(hp.Float('dropout3', 0.1, 0.5, step=0.1)))

    model.add(Dense(
        hp.Int('dense_units1', min_value=16, max_value=64, step=8),
        activation='relu'
    ))
    model.add(Dropout(hp.Float('dropout4', 0.1, 0.5, step=0.1)))

    # BUG FIX: Added model.add() here. Previously these were disconnected.
    model.add(Dense(hp.Int('dense_units2', min_value=16, max_value=32, step=4), activation='relu'))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(9, activation='softmax')) # Hardcoded 9 for safety, replace with y_train_seq_final.shape[1]

    gamma = hp.Float('gamma', 1.0, 4.0, step=0.5)
    alpha = hp.Float('alpha', 0.1, 0.5, step=0.05)
    learning_rate = hp.Float('learning_rate', 1e-4, 5e-3, sampling='LOG')

    lr_schedule = CosineDecayRestarts(initial_learning_rate=learning_rate, first_decay_steps=5)
    optimizer = Adam(learning_rate=lr_schedule, clipnorm=1.0)

    # Note: You need to make sure 'focal_loss' and 'custom_objective' are defined in your scope
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=gamma, alpha=alpha),
        metrics=['accuracy', custom_objective, tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model
