
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transfroms
#from resnet50 import Resnet50
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models




if torch.cuda.is_available():
     device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = models.resnet50(pretrained= True)
in_channel = model.fc.in_features
model.fc=nn.Linear(in_channel, 10)
model =model.to(device)
#model = Resnet50().to(device)
torch.cuda.empty_cache()
print(device,' ', torch.cuda.is_available())


batchsize = 8

transfrom_train = transfroms.Compose([
    transfroms.Resize(224),
    transfroms.RandomHorizontalFlip(0.5),
    transfroms.ToTensor(),
    transfroms.Normalize(mean=(0.488, 0.481, 0.445),
                         std=(0.248, 0.245, 0.266))
])



cifar10 = torchvision.datasets.CIFAR10(
    root='datasets',
    train=True,
    transform=transfrom_train,
    download=False
)



cifar10_train, cifar10_val = torch.utils.data.random_split(cifar10, [45000, 5000], generator=torch.Generator().manual_seed(0))

data_train = torch.utils.data.DataLoader(cifar10_train, 64, True)
data_val = torch.utils.data.DataLoader(cifar10_val, 64, True)





criteon = nn.CrossEntropyLoss().to(device)
lr_initial = 0.001
optimizer = optim.SGD(model.parameters(), lr=lr_initial, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma = 0.5)

for epoch in range(30):
    model.train()
    for i, (x, label) in enumerate(data_train):
        x = x.to(device)
        label = label.to(device)

        logits = model(x)

        loss = criteon(logits,label)
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    print(epoch, 'lr:',optimizer.param_groups[0]['lr'])
    print(epoch, 'train loss', loss.item())
    scheduler.step()

   
    model.eval()
    with torch.no_grad():
        total_correct = 0;
        total_num = 0;
        for i, (x, label) in enumerate(data_val):
            x, label = x.to(device), label.to(device)

            logits = model(x)
            
            loss = criteon(logits,label)
            pred = logits.argmax(dim=1)

            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)

        accurancy = total_correct/total_num
        print(epoch, 'val loss', loss.item())
        print(epoch, 'validation accurancy: ', accurancy)

#torch.save(model, 'model.pth')

