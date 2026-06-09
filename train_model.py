from utils.preprocessing import get_dataloader
from utils.train import build_model,train_model

dataset_path='E:\desktop data\mentorship program devsil\Currency_Note_Recongnition\dataset'
model_save='models/model.keras'
epochs=15
batch_size=32

print("Loading Dataset....")

train_data,val_data,class_names=get_dataloader(dataset_path,batch_size)

print("model Building...")

model,base_model=build_model(num_classes=len(class_names))

print("Training Model...")

model.history=train_model(
    model=model,
    base_model=base_model,
    train_data=train_data,
    val_data=val_data,
    class_names=class_names,
    epochs=epochs,
    save_path=model_save
)
print("Done!")