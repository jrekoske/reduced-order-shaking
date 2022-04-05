import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Reshape, Conv2DTranspose, LeakyReLU, Conv2D
from keras.optimizers import Adam
from keras.utils import plot_model
from tqdm.keras import TqdmCallback
from sklearn.utils import shuffle

dep_min = 0
dep_max = 60  # km
strike_min = 0
strike_max = 360  # degrees

deps = np.linspace(dep_min, dep_max, 100)
strikes = np.linspace(strike_min, strike_max, 100)
dep_grid, strike_grid = np.meshgrid(deps, strikes)
grid = np.column_stack((dep_grid.ravel(), strike_grid.ravel()))

shakes = np.load('shakes_32.npy')
shakes = shakes.reshape((10000, 32, 32))

grid, shakes = shuffle(grid, shakes, random_state=0)
N = 1000
grid, shakes = grid[:N], shakes[:N]

model = Sequential()
model.add(Dense(8 * 8 * 8, input_dim=2))
model.add(LeakyReLU(alpha=0.2))
model.add(Reshape((8, 8, 8)))
model.add(Conv2DTranspose(8, (4, 4), strides=(2, 2), padding='same'))
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2DTranspose(4, (4, 4), strides=(2, 2), padding='same'))
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(1, (7, 7), padding='same'))

plot_model(model, show_shapes=True)

opt = Adam(learning_rate=0.01)
model.compile(loss='mae', optimizer=opt)
history = model.fit(grid, shakes, epochs=200, validation_split=0.25, verbose=0, callbacks=[TqdmCallback(verbose=0)])

plt.plot(history.history['val_loss'], label='val loss')
plt.plot(history.history['loss'], label='loss')
plt.yscale('log')
plt.legend()
plt.savefig('loss.pdf')
plt.close('all')

true = shakes[0]
pred = model.predict([[grid[0][0], grid[0][1]]]).reshape(32, 32)

plt.figure(figsize=(7, 2.5))
plt.subplot(1, 2, 1)
plt.imshow(true)
plt.title('True')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(pred)
plt.colorbar()
plt.title('Predicted')
plt.tight_layout()
plt.savefig('pred.pdf')
plt.close('all')
