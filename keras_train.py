"""
Author: Jason Eisele
Email: jeisele@shipt.com
Date: August 1, 2020
"""
import argparse
import keras
import tensorflow as tf
import cloudpickle

parser = argparse.ArgumentParser(
  description='Train a Keras feed-forward network for MNIST classification')
parser.add_argument('--batch-size', '-b', type=int, default=128)
parser.add_argument('--epochs', '-e', type=int, default=1)
parser.add_argument('--learning_rate', '-l', type=float, default=0.05)
parser.add_argument('--num-hidden-units', '-n', type=float, default=512)
parser.add_argument('--dropout', '-d', type=float, default=0.25)
parser.add_argument('--momentum', '-m', type=float, default=0.85)
args = parser.parse_args()

mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

model = keras.models.Sequential([
  keras.layers.Flatten(input_shape=X_train[0].shape),
  keras.layers.Dense(args.num_hidden_units, activation=tf.nn.relu),
  keras.layers.Dropout(args.dropout),
  keras.layers.Dense(10, activation=tf.nn.softmax)
])

optimizer = keras.optimizers.SGD(lr=args.learning_rate, momentum=args.momentum)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
