import tensorflow as tf

def get_dataloader(dataset_path,batch_size=32,image_size=(224,224)):
    
    # loading training data
    """
    read image from path and decode it to tensor
    folder name become the label of the image
    data_set afghani become 0
    data_set PKR becom 1
    """

    train_data=tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset='training',
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    val_data=tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset='validation',
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )


    # get class name from folder name
    class_names=train_data.class_names
    print(f"classes found : {class_names}")

    #Augmentaion for training
    #concept: Augmentation 
    #makes model see more variety during training 
    #prevent overfitting on small dataset
    #only applied during training not validation

    augmentation=tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1)
            ])

    #Normalization
    # preprocess input
    #trained dATA IS DIVIDE BY 255
    def preprocess_train(images,labels):
        images=augmentation(images,trainig=True)
        images=tf.keras.applications.resnet50.preprocess_input(images)
        return images,labels

    def preprocess_val(images,labels):
        images=tf.keras.applications.resnet50.preprocess_input(images)
        return images,labels

    # apply preprocessing to training and validation data
    train_data=train_data.map(preprocess_train,num_parallel_calls=tf.data.AUTOTUNE)
    val_data=val_data.map(preprocess_val,num_parallel_calls=tf.data.AUTOTUNE)

    #catche and prefetch for performance AND SPEED
    train_data=train_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_data=val_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    print("Preprocessing file running successfully")
    return train_data,val_data,class_names