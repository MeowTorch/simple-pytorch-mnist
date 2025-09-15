# train.py
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

# 设置中文字体
plt.rcParams['font.family'] = 'SimHei'

# 定义神经网络结构
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 输入层 -> 第一隐藏层
        self.fc1 = torch.nn.Linear(28*28, 128)
        # 第一隐藏层 -> 第二隐藏层
        self.fc2 = torch.nn.Linear(128, 128)
        # 第二隐藏层 -> 第三隐藏层
        self.fc3 = torch.nn.Linear(128, 64)
        # 第三隐藏层 -> 输出层
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x

# 数据加载函数
def get_data_loader(is_train, batch_size=64):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST(
        root="", train=is_train, transform=to_tensor, download=True
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)

# 模型评估函数
def evaluate(test_data, net, device):
    net.eval()
    n_correct, n_total = 0, 0
    with torch.no_grad():
        for x, y in test_data:
            x, y = x.to(device), y.to(device)
            output = net(x.view(-1, 28*28))
            pred = torch.argmax(output, dim=1)
            n_correct += (pred == y).sum().item()
            n_total += y.size(0)
    return n_correct / n_total

def main():
    print("📥 正在加载 MNIST 数据集 ...")
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    print("✅ 数据加载完成！")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)

    print(f"🔍 初始模型准确率: {evaluate(test_data, net, device)*100:.2f}%")

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    num_epochs = 15
    print("🚀 开始训练模型 ...")

    for epoch in range(num_epochs):
        net.train()
        for batch_idx, (x, y) in enumerate(train_data, start=1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = net(x.view(-1, 28*28))
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"📊 [第{epoch+1}轮 | 批次 {batch_idx}] 当前损失: {loss.item():.4f}")

        acc = evaluate(test_data, net, device)
        print(f"✅ 第 {epoch+1} 轮训练结束，测试集准确率: {acc*100:.2f}%")

    torch.save(net.state_dict(), "mnist_net.pth")
    print("✅ 模型已保存为 mnist_net.pth")

    # 展示前4张测试集样本及预测结果
    print("\n📸 展示测试集样本预测结果:")
    for n, (x, y) in enumerate(test_data):
        if n > 3:
            break
        x0 = x[0].to(device)
        predict = torch.argmax(net(x0.view(-1, 28*28))).item()
        plt.figure(n)
        plt.imshow(x[0].view(28,28), cmap="gray")
        plt.title(f"预测结果: {predict} | 实际标签: {y[0].item()}")
    plt.show()
    print("🎉 训练完成！")

if __name__ == "__main__":
    main()
