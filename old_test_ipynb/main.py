import torch as t
import torch.nn as nn
import numpy as np 
import random
import dataset_utils
import emotion_detection.model_utils.model_utils as model_utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def seed_everything(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.benchmark = False
    t.backends.cudnn.deterministic = True

def train(model,train_loader,optimizer,criterion,num_epochs,device):
    loss_arr = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            # Forward pass
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_arr.append(total_loss/len(train_loader))
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {total_loss/len(train_loader)}')
    return loss_arr
        
def test(model,test_loader,criterion,device):
    
    def accuracy(y_true, y_pred):
        eq = t.eq(y_true, y_pred).int()
        return sum(eq)/len(eq)

    with t.no_grad():
        model.eval()
        for inputs,labels in test_loader:
            outputs = model(inputs.to(device))
            outputs1 = outputs.detach().cpu()
            acc += accuracy(labels,outputs1)
        print(f"accuracy: {(acc/len(test_loader))*100: 0.2f}%")

def save_model(name,model):
    t.save(model.state_dict(), f'models/{name}.pth')
    
def plot_graph(arr,epochs):
    plt.title("Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(arr,range(1,epochs+1))
    plt.show()
    

seed_everything()
audio_paths = ["path/to/audio1.wav", "path/to/audio2.wav", ...]
labels = [0, 1, ...] 


train_dataset, test_dataset = train_test_split(combined_datasets, test_size=0.2, random_state=54)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

dataset = dataset_utils.AudioDataset(audio_paths, labels)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



num_classes = 7 
model = model_utils.EmotionCNN(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr=0.001)