import torch
import torchvision
import torchvision.transforms as transfroms


if torch.cuda.is_available():
     device = torch.device('cuda')
else:
    device = torch.device('cpu')

transfrom_test = transfroms.Compose([
    transfroms.Resize(224),
    transfroms.RandomHorizontalFlip(),
    transfroms.ToTensor(),
    transfroms.Normalize(mean=(0.474, 0.473, 0.430),
                         std=(0.255, 0.252, 0.269))
])

cifar10_test = torchvision.datasets.CIFAR10(
    root='datasets',
    train=False,
    transform=transfrom_test,
    download=False
)
data_test = torch.utils.data.DataLoader(cifar10_test, 64, True)

model=torch.load('model.pth')
model=model.to(device)




model.eval()
with torch.no_grad():
        total_correct = 0;
        total_num = 0;
        for x, label in data_test:
            x, label = x.to(device), label.to(device)

            logits = model(x)
            
            pred = logits.argmax(dim=1)

            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)

        accurancy = total_correct/total_num
        print('test accurancy: ', accurancy)