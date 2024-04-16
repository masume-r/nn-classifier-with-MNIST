def train(model, trn_loader, device, criterion, optimizer):
    model.train()  
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in trn_loader:
        images, labels = images.to(device), labels.to(device)  
        optimizer.zero_grad()  
        outputs = model(images)  
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step()  
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    trn_loss = running_loss / len(trn_loader.dataset)
    acc = correct / total
    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    model.eval()  
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tst_loader:
            images, labels = images.to(device), labels.to(device)  
            outputs = model(images)  
            loss = criterion(outputs, labels)  
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    tst_loss = running_loss / len(tst_loader.dataset)
    acc = correct / total
    return tst_loss, acc

def plot_curves(train_losses, test_losses, train_accuracies, test_accuracies, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss Curves')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Accuracy Curves')
    plt.legend()
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    
    train_dataset = MNIST('/content/train.tar', transform=transform)
    test_dataset = MNIST('/content/test.tar', transform=transform)

    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    
    lenet_model = LeNet5().to(device)  
    mlp_model = CustomMLP().to(device)  
    optimizer = optim.SGD(lenet_model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    lenet_train_losses, lenet_test_losses = [], []
    lenet_train_accuracies, lenet_test_accuracies = [], []
    mlp_train_losses, mlp_test_losses = [], []
    mlp_train_accuracies, mlp_test_accuracies = [], []

    
    for epoch in range(10):  
        train_loss, train_acc = train(lenet_model, train_loader, device, criterion, optimizer)
        test_loss, test_acc = test(lenet_model, test_loader, device, criterion)
        print(f"Epoch {epoch+1}: LeNet5 Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}%, Test Loss {test_loss:.4f}, Test Acc {test_acc:.2f}%")
        lenet_train_losses.append(train_loss)
        lenet_test_losses.append(test_loss)
        lenet_train_accuracies.append(train_acc)
        lenet_test_accuracies.append(test_acc)

    
    optimizer = optim.SGD(mlp_model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(10):  
        train_loss, train_acc = train(mlp_model, train_loader, device, criterion, optimizer)
        test_loss, test_acc = test(mlp_model, test_loader, device, criterion)
        print(f"Epoch {epoch+1}: CustomMLP Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}%, Test Loss {test_loss:.4f}, Test Acc {test_acc:.2f}%")
        mlp_train_losses.append(train_loss)
        mlp_test_losses.append(test_loss)
        mlp_train_accuracies.append(train_acc)
        mlp_test_accuracies.append(test_acc)

    
    plot_curves(lenet_train_losses, lenet_test_losses, lenet_train_accuracies, lenet_test_accuracies, 'LeNet5')
    plot_curves(mlp_train_losses, mlp_test_losses, mlp_train_accuracies, mlp_test_accuracies, 'CustomMLP')

if __name__ == '__main__':
    main()
