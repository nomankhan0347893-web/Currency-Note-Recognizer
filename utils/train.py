import tensorflow as tf
import json
import os

def build_model(num_classes=2):
    """
    Load the ResNet50 model without the top layer and add a new classification head
    Freeze the base model layers to prevent them from being updated during training
    Add our layers on top of the base model
    """
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False # Freeze the base model layers

    # adding classification layers on top of the base model
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

    
    #compile the model with an appropriate loss function and optimizer
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    return model, base_model

def train_model(model,base_model,train_data,val_data,class_names,epochs=15,save_path='models/model.keras'):

    # callbacks for saving the best model and early stopping
    callbacks = [
        # save the best model based on validation accuracy
        tf.keras.callbacks.ModelCheckpoint(save_path, save_best_only=True, monitor='val_accuracy', verbose=1),
        # stop training if validation accuracy does not improve for 5 consecutive epochs
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1,restore_best_weights=True),
        # reduce learning rate if validation accuracy does not improve for 3 consecutive epochs
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, verbose=1)
    ]
    # Train top layers
    print("Training top layers...")
    history1=model.fit(train_data, validation_data=val_data, epochs=epochs//2, callbacks=callbacks)

    # Fine tune the model by unfreezing some of the base model layers
    print("Fine-tuning the model...")
    base_model.trainable = True # Unfreeze the base model layers

    # only unfreeze the last 30 layers of the base model
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    # recompile the model with a lower learning rate for fine-tuning
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
    )
    history2=model.fit(
        train_data,
        epochs=epochs//2,
        validation_data=val_data,
        callbacks=callbacks
    )             

    #save class name
    json.dump(
        {'class_name':class_names},
        open('models/class_names.json','w')
    )
    # save_combine hsitory
    history={
        'accuracy' :history1.history['accuracy']+
                    history2.history['accuracy'],
        'loss'     :history1.history['loss']+
                    history2.history['loss'],
        'val_accuracy':history1.history['val_accuracy']+
                       history2.history['val_accuracy'],
        'val_loss'  :history1.history['val_loss']+
                     history2.history['val_loss']
    }
    json.dump(history,open('models/history.json','w'))
    best_acc=max(history['val_accuracy'])
    print(f"\nBest val accuracy: {best_acc*100:.2f}%")

    return model,history