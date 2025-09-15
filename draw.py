# draw.py
import tkinter as tk

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw


# 定义网络（要和 train.py 里的结构保持一致）
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DrawBoard:
    def __init__(self, net, size=280):
        self.size = size
        self.net = net.eval()

        self.root = tk.Tk()
        self.root.title("手写数字识别")

        # 画布
        self.canvas = tk.Canvas(self.root, width=size, height=size, bg="black")
        self.canvas.pack()

        # 鼠标拖动画画
        self.canvas.bind("<B1-Motion>", self.paint)

        # 按钮
        tk.Button(self.root, text="清空", command=self.clear).pack(side=tk.LEFT)
        tk.Button(self.root, text="识别", command=self.predict).pack(side=tk.RIGHT)

        # 用 Pillow 存储图像
        self.image = Image.new("L", (size, size), 0)
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x, y = event.x, event.y
        r = 10  # 画笔半径
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.draw.ellipse((x-r, y-r, x+r, y+r), fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, self.size, self.size), fill=0)

    def predict(self):
        # 处理图像
        img = self.image.resize((28, 28))
        # img.show()
        img_data = np.array(img) / 255.0
        x = torch.tensor(img_data, dtype=torch.float32).view(-1, 28*28)
        x = x.to(next(self.net.parameters()).device)

        # 模型预测
        with torch.no_grad():
            output = self.net(x)
            prob = torch.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1).item()
            confidence = prob[0, pred].item() * 100

        print(f"👉 识别结果: {pred} (置信度 {confidence:.2f}%)")
        self.root.title(f"识别结果: {pred} ({confidence:.1f}%)")

    def run(self):
        self.root.mainloop()


def main():
    # 加载模型
    net = Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load("mnist_net.pth", map_location=device)
    net.load_state_dict(state_dict)   # 加载参数
    net.to(device).eval()

    # 打开手写板
    board = DrawBoard(net)
    board.run()


if __name__ == "__main__":
    main()
