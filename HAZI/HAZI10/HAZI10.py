# %%
import tensorflow as tf
import matplotlib.pyplot as plt

# %%
'''
Készíts egy metódust ami a mnist adatbázisból betölti a train és test adatokat. (tf.keras.datasets.mnist.load_data())
Majd a tanitó, és tesztelő adatokat normalizálja, és vissza is tér velük.


Egy példa a kimenetre: train_images, train_labels, test_images, test_labels
függvény neve: mnist_digit_data
'''

# %%
def mnist_digit_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    #train_labels = train_labels / 10
    #test_labels = test_labels / 10
    return (train_images, train_labels, test_images, test_labels)

    """
    print(train_images.shape)
    print(len(train_labels))
    print(train_labels)
    print(test_images.shape)
    print(len(test_labels))
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()
    """ 

datas = mnist_digit_data()

# %%
'''
Készíts egy neurális hálót, ami képes felismerni a kézírásos számokat.
A háló kimenete legyen 10 elemű, és a softmax aktivációs függvényt használja.
Hálon belül tetszőleges számú réteg lehet.


Egy példa a kimenetre: model,
return type: keras.engine.sequential.Sequential
függvény neve: mnist_model
'''

# %%
from tensorflow.keras.models import Sequential

def mnist_model() -> Sequential:
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='softmax'),
        tf.keras.layers.Dense(10)
    ])

    return model

model = mnist_model()
print(model)

# %%
'''
Készíts egy metódust, ami a bemeneti hálot compile-olja.
Optimizer: Adam
Loss: SparseCategoricalCrossentropy(from_logits=False)

Egy példa a bemenetre: model
Egy példa a kimenetre: model
return type: keras.engine.sequential.Sequential
függvény neve: model_compile
'''

# %%
def model_compile(model : Sequential) -> Sequential:
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    return model

print(model_compile(model))

# %%
'''
Készíts egy metódust, ami a bemeneti hálót feltanítja.

Egy példa a bemenetre: model,epochs, train_images, train_labels
Egy példa a kimenetre: model
return type: keras.engine.sequential.Sequential
függvény neve: model_fit
'''

# %%
def model_fit(model,epochs, train_images, train_labels):
    model.fit(train_images, train_labels, epochs=epochs)
    return model

model_fit(model, 4, datas[0], datas[1])

# %%
'''
Készíts egy metódust, ami a bemeneti hálót kiértékeli a teszt adatokon.

Egy példa a bemenetre: model, test_images, test_labels
Egy példa a kimenetre: test_loss, test_acc
return type: float, float
függvény neve: model_evaluate
'''

# %%
def model_evaluate(model, test_images, test_labels):
    return model.evaluate(test_images,  test_labels)

test_loss, test_acc = model_evaluate(model, datas[2], datas[3])
print('\nTest loss:', test_loss)
print('\nTest accuracy:', test_acc)

# %%



