# Proyecto Final Curso IA

### Descripción

En este proyecto entrenaremos un modelo de clasificación de tumores cerebrales observando imágenes de resonancias magnéticas.

-   Modelo: `Sequential`
-   Dataset: > 3000 imágenes de resonancias magnéticas clasificadas en grupos: `normal`, `pituitary_tumor`, `meningioma_tumor`, `glioma_tumor`

### Importaciones

```python
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models, layers , optimizers
import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.utils import to_categorical, plot_model
import random
from tensorflow.keras.preprocessing import image
```

Definimos la variable `data_path` en la que almacenamos la ruta de nustro dataset

```python
data_path = '/kaggle/input/brain-tumors-256x256/Data'
```

Definifos una función `create_dataframe` que recibe una ruta, itera por los directorios que hay en esa ruta, y por cada directorio itera por cada una de las imágenes, devolviendo las rutas de cada una de las imágenes con una etiqueta, que en este caso es el nombre del directorio que lo contiene

```python
def create_dataframe(data_path):
    filepaths = []
    labels = []

    for fold in os.listdir(data_path):
        f_path = os.path.join(data_path, fold)
        if os.path.isdir(f_path):
            imgs = os.listdir(f_path)
            for img in imgs:
                img_path = os.path.join(f_path, img)
                filepaths.append(img_path)
                labels.append(fold)

    fseries = pd.Series(filepaths, name='Filepaths')
    lseries = pd.Series(labels, name='Labels')
    return pd.concat([fseries, lseries], axis=1)

df = create_dataframe(data_path)
df
```

![imágenes etiquetadas](image.png)

### División del dataset en subgrupos

A continuación dividimos nuestro conjunto de datos en tres subconjuntos: **entrenamiento**, **validación** y **prueba**, de manera aleatoria.

Haremos una primera división:

-   **`train_df`**: contiene el 70% de los datos, y se usará para el entrenamiento del modelo.
-   **`dummy_df`**: contiene el 30% restante de los datos y se usará para dividirlo más adelante en conjuntos de validación y prueba.

De ese 30% restante, `dummy_df`, haremos una segunda división:

-   **`test_df`**: recibe el 2/3 del `dummy_df`, que equivale a un **20%** del conjunto de datos original.

-   **`valid_df`**: recibe el 1/3 restante de `dummy_df`, que equivale al **10%** del conjunto de datos original.

Iprimimos por pantalla las dimensiones de las diferentes muestras, para saber cuántos elementos tenemos:
Nos da dos valores por cada subconjunto, el número de datos (filas) y el número de columnas que hay en cada conjunto (uno con el nombre, y otro con la etiqueta)

```python
print(train_df.shape)  // (2167, 2)
print(dummy_df.shape)  // (929, 2)
print(valid_df.shape)  // (310, 2)
print(test_df.shape)  // (619, 2)
```

Si hacemos la suma nos da el valor total del dataset original, lo que nos ayuda a confirmar que todos los elementos han sido tenidos en cuenta

A continuación reescalamos las imágenes para entrenar el modelo de forma más eficiente

```python
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
```

Generamos lotes de imágenes redimensionadas a 224x224 píxeles, junto con sus etiquetas

```python
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepaths',
    y_col='Labels',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_df,
    x_col='Filepaths',
    y_col='Labels',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepaths',
    y_col='Labels',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
```

![alt text](image-1.png)

Definimos y ejecutamos una función que nos muestra cada una de las imágenes del subconjunto que le mandamos junto con su etiqueta

```python
def show_images(image_gen):
    class_dict = image_gen.class_indices
    classes = list(class_dict.keys())

    images, labels = next(image_gen)

    plt.figure(figsize=(20, 20))

    num_images = min(len(labels), 25)

    for i in range(num_images):
        plt.subplot(5, 5, i + 1)

        image = images[i]

        plt.imshow(image)

        index = np.argmax(labels[i])
        class_name = classes[index]

        plt.title(class_name, color="green", fontsize=20)
        plt.axis('off')

    plt.show()

show_images(train_generator)
```

> En este caso le pasamos el subconjunto `train_generator` que obtuvimos con el bloque de código anterior pasándole `train_datagen`. Podríamos haber hecho lo mismo con `valid_generator` y `test_generator`

![alt text](image-2.png)

### Modelo

Ahora difinimos un modelo de red neuronal convolucional (CNN) en Keras utilizando la API `Sequential`. El objetivo es crear un modelo para tareas de clasificación de imágenes, con varias capas convolucionales.

Tendremos:

-   **5 capas convolucionales** para extraer características de las imágenes.
-   **Capas de agrupamiento (MaxPooling)** para reducir las dimensiones de los datos.
-   Una **capa Flatten** para convertir los mapas de características en un vector 1D.
-   Una **capa densa** de 256 unidades con activación **ReLU**, seguida de **Dropout** para evitar el sobreajuste.
-   Una **capa de salida** con activación **softmax** para clasificar en una de las clases.

```python
input_shape = (224, 224, 3)
n_classes = len(train_generator.class_indices)

model = keras.Sequential([
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    keras.layers.MaxPool2D(pool_size=(2, 2)),

    keras.layers.Conv2D(filters=128, kernel_size=(5, 5), activation='relu', padding='same'),
    keras.layers.MaxPool2D(pool_size=(3, 3)),

    keras.layers.Conv2D(filters=128, kernel_size=(7, 7), activation='relu', padding='same'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),

    keras.layers.Conv2D(filters=128, kernel_size=(5, 5), activation='relu', padding='same'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),

    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(n_classes, activation='softmax')
])


model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.optimizers.Adam(),
    metrics=['accuracy']
)

model.summary()
```

Este resumen detalla la arquitectura del modelo **CNN** en Keras, mostrando información clave sobre cada capa.

#### Columnas del resumen:

-   **Layer (type)**: El nombre y tipo de cada capa en el modelo.
-   **Output Shape**: El tamaño de las salidas después de que los datos pasan por esa capa.
-   **Param #**: El número de parámetros entrenables en esa capa.

![alt text](image-3.png)

#### Entrenamiento del modelo

```python
history = model.fit(
    train_generator,
    batch_size=32,
    validation_data=valid_generator,
    epochs=15
)
```

![alt text](image-4.png)

> El entrenamiento realizado ha sido reducido en valores de epochs debido a la falta de potencia de mi ordenador, se puede ver que con apenas 15 epochs ha tardado más de 2 horas.

Evaluamos el modelo

```python
scores = model.evaluate(test_generator)
scores
```

![alt text](image-5.png)

Realizamos la predicción sobre el conjunto de datos de prueba utilizando el modelo y luego evaluamos el rendimiento del modelo en términos de métricas de clasificación detalladas.

```python
test_generator.reset()
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_generator.classes

class_labels = list(test_generator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)
```

![alt text](image-6.png)

Generamos una matriz de confusión para evaluar el rendimiento de tu modelo en el conjunto de datos de prueba

```python
predictions = model.predict(test_generator)

predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_generator.classes

cm = confusion_matrix(true_classes, predicted_classes)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_indices)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
```

Cada celda en la matriz de confusión muestra el número de ejemplos predichos para cada combinación de clase verdadera y clase predicha. Las diagonales representan las predicciones correctas, mientras que las celdas fuera de la diagonal representan errores de clasificación.

![alt text](image-7.png)

Ahora trazamos dos gráficos para visualizar el desempeño del modelo durante el entrenamiento. Uno muestra la precisión (accuracy) y el otro muestra la pérdida (loss) en función de las épocas

### Gráfico de Precisión

![alt text](image-8.png)

-   **Precisión en Entrenamiento:** Muestra cómo la precisión del modelo en el conjunto de entrenamiento cambia con el tiempo. Idealmente, debería aumentar y estabilizarse.
-   **Precisión en Validación:** Muestra la precisión en el conjunto de validación. Se espera que también aumente, pero no siempre debe seguir la misma tendencia que la precisión en el entrenamiento.

### Gráfico de Pérdida

![alt text](image-9.png)

-   **Pérdida en Entrenamiento:** Muestra cómo disminuye la pérdida en el conjunto de entrenamiento. Idealmente, debería disminuir con el tiempo.
-   **Pérdida en Validación:** Muestra la pérdida en el conjunto de validación. Debería disminuir, pero un aumento en la pérdida de validación puede indicar sobreajuste (overfitting).

Ponemos a prueba el modelo con 6 imágenes aleatorias del conjunto `test_df` y devolvemos la imagen con su etiqueta correcta y la etiqueta que predice el modelo

```python
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

class_labels = list(train_generator.class_indices.keys())

def predict_random_images(test_df, model, class_labels, num_images=6):
    random_images = test_df.sample(n=num_images).reset_index(drop=True)
    plt.figure(figsize=(15, 10))

    for i in range(num_images):
        img_path = random_images.loc[i, 'Filepaths']
        actual_label = random_images.loc[i, 'Labels']
        img_array = load_and_preprocess_image(img_path)

        prediction = model.predict(img_array)
        predicted_label = class_labels[np.argmax(prediction)]

        plt.subplot(2, 3, i+1)
        img = image.load_img(img_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Actual: {actual_label}\nPredicted: {predicted_label}")

    plt.show()

predict_random_images(test_df, model, class_labels, num_images=6)
```

![alt text](image-10.png)

Observamos que en algunos casos falla y realiza una predicción errónea>
