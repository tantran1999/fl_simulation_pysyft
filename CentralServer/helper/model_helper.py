import syft as sy


class FashionMNIST_TEST(sy.Module):
    def __init__(self, torch_ref):
        super().__init__(torch_ref=torch_ref)
        self.layer1 = self.torch_ref.nn.Conv2d(1, 4, 3, 1, 1)
        self.layer2 = self.torch_ref.nn.Conv2d(4, 8, 3, 1, 1)
        self.fc1 = self.torch_ref.nn.Linear(in_features=7*7*8, out_features=200)
        self.drop = self.torch_ref.nn.Dropout2d(0.25)
        self.fc2 = self.torch_ref.nn.Linear(in_features=200, out_features=10)

    def forward(self, X):
        out = self.layer1(X)
        out = self.torch_ref.nn.functional.relu(out)
        out = self.torch_ref.nn.functional.max_pool2d(out, 2, 2)
        out = self.layer2(out)
        out = self.torch_ref.nn.functional.relu(out)
        out = self.torch_ref.nn.functional.max_pool2d(out, 2, 2)
        out = out.view(-1, 7*7*8)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.torch_ref.nn.functional.relu(out)
        out = self.fc2(out)
        return out
# Define models like Data poisoning paper
# CNN model for FashionMNIST images with size 28*28 and gray (1 channel)
# CNN model for CIFAR10 images with size size 32*32 and RGB (3 channel)
# For experiment
class FashionMNIST(sy.Module):
    def __init__(self, torch_ref):
        super().__init__(torch_ref=torch_ref)
        self.conv1 = self.torch_ref.nn.Conv2d(1, 16, kernel_size=5, padding=2)
        #self.bn1 = self.torch_ref.nn.BatchNorm2d(16)

        self.conv2 = self.torch_ref.nn.Conv2d(16, 32, kernel_size=5, padding=2)
        #self.bn2 = self.torch_ref.nn.BatchNorm2d(32)

        self.fc1 = self.torch_ref.nn.Linear(7*7*32, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.torch_ref.nn.functional.relu(x)
        # x = self.bn1(x)
        x = self.torch_ref.nn.functional.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = self.torch_ref.nn.functional.relu(x)
        # x = self.bn2(x)
        x = self.torch_ref.nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 7*7*32)
        x = self.fc1(x)
        return x

# For experiment
class CIFAR10(sy.Module):
    def __init__(self, torch_ref):
        super().__init__(torch_ref=torch_ref)
        self.conv1 = self.torch_ref.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = self.torch_ref.nn.BatchNorm2d(32)
        self.conv2 = self.torch_ref.nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = self.torch_ref.nn.BatchNorm2d(32)

        self.conv3 = self.torch_ref.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = self.torch_ref.nn.BatchNorm2d(64)
        self.conv4 = self.torch_ref.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = self.torch_ref.nn.BatchNorm2d(64)

        self.conv5 = self.torch_ref.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = self.torch_ref.nn.BatchNorm2d(128)
        self.conv6 = self.torch_ref.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = self.torch_ref.nn.BatchNorm2d(128)

        self.fc1 = self.torch_ref.nn.Linear(128 * 4 * 4, 128)
        self.fc2 = self.torch_ref.nn.Linear(128, 10)

    def forward(self, x):
        x = self.bn1(self.torch_ref.nn.functional.relu(self.conv1(x)))
        x = self.bn2(self.torch_ref.nn.functional.relu(self.conv2(x)))
        x = self.torch_ref.nn.functional.max_pool2d(x, 2, 2)

        x = self.bn3(self.torch_ref.nn.functional.relu(self.conv3(x)))
        x = self.bn4(self.torch_ref.nn.functional.relu(self.conv4(x)))
        x = self.torch_ref.nn.functional.max_pool2d(x, 2, 2)

        x = self.bn5(self.torch_ref.nn.functional.relu(self.conv5(x)))
        x = self.bn6(self.torch_ref.nn.functional.relu(self.conv6(x)))
        x = self.torch_ref.nn.functional.max_pool2d(x, 2, 2)

        x = x.view(-1, 128 * 4 * 4)

        x = self.fc1(x)
        x = self.torch_ref.nn.functional.softmax(self.fc2(x))
        return x
