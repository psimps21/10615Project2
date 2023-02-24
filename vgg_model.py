import torch
import numpy as np
import torchvision.transforms
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm
import ssl
import pickle


def labels2onehot(labels):
    return np.array([[i == lab for i in range(20)] for lab in labels])


class AgnosticModel(torch.nn.Module):
    def __init__(self, base_model, num_classes=5):
        super(AgnosticModel, self).__init__()

        self.norm_layer = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.base_model = base_model
        self.act_layer = torch.nn.Softmax(dim=1)

    def forward(self, imgs):
        norm_imgs = self.norm_layer(imgs)
        raw_preds = self.base_model(norm_imgs)
        preds = self.act_layer(raw_preds)

        return preds


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = torchvision.models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = torchvision.models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = torch.nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = torchvision.models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = torch.nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = torchvision.models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = torchvision.models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = torchvision.models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = torch.nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def training_loop_w_pretrain(data_fname, labels_fname, epochs, model):
    samples = np.load(data_fname)
    labels = np.load(labels_fname)

    samples = samples / 255.0

    p = np.random.permutation(len(samples))

    samples = samples[p]
    labels = labels[p]

    train_test_split = int(samples.shape[0]*0.9)

    train = []
    for i in range(train_test_split):
        train.append([samples[i], labels[i]])

    test = []
    for i in range(train_test_split, len(samples)):
        test.append([samples[i], labels[i]])

    batch_size = 16
    learning_rate = 0.005

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)
    acc = MulticlassAccuracy(num_classes=5)

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=train,
                                              batch_size=batch_size,
                                              shuffle=True)

    test_accs = []

    for epoch in tqdm(range(epochs)):
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.type(torch.float32)

            imgs = torch.swapaxes(imgs, 1, 3)
            pred = model(imgs)

            l = loss(pred, labels)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        model.eval()
        i = 0
        epoch_acc = 0
        for imgs, labels in test_loader:
            imgs = imgs.type(torch.float32)
            imgs = torch.swapaxes(imgs, 1, 3)
            pred = torch.argmax(model(imgs), dim=1)

            batch_acc = acc(pred, labels)
            epoch_acc += batch_acc.item()
            i += 1

        test_accs.append(epoch_acc/i)
        print(epoch_acc/i)
        optimizer.zero_grad()

    return test_accs, model


def training_loop(data_fname, labels_fname, epochs):
    samples = np.load(data_fname)
    labels = np.load(labels_fname)

    samples = samples / 255.0

    p = np.random.permutation(len(samples))

    samples = samples[p]
    labels = labels[p]

    train_test_split = int(samples.shape[0]*0.9)

    train = []
    for i in range(train_test_split):
        train.append([samples[i], labels[i]])

    test = []
    for i in range(train_test_split, len(samples)):
        test.append([samples[i], labels[i]])

    num_epochs = 20
    batch_size = 16
    learning_rate = 0.005

    model = VGG()

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)
    acc = MulticlassAccuracy(num_classes=5)

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test,
                                              batch_size=batch_size,
                                              shuffle=True)

    test_accs = []
    pred_dict = {i: 0 for i in range(5)}

    for epoch in tqdm(range(epochs)):
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.type(torch.float32)

            imgs = torch.swapaxes(imgs, 1, 3)
            pred = model(imgs)

            l = loss(pred, labels)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        model.eval()
        i = 0
        epoch_acc = 0
        for imgs, labels in test_loader:
            imgs = imgs.type(torch.float32)

            imgs = torch.swapaxes(imgs, 1, 3)

            pred = torch.argmax(model(imgs), dim=1)

            batch_acc = acc(pred, labels)
            epoch_acc += batch_acc.item()
            i += 1

            pred_np = pred.detach().numpy()

            for val in pred_np:
                pred_dict[int(val)] += 1

        test_accs.append(epoch_acc/i)
        print(epoch_acc/i)
        optimizer.zero_grad()

    return test_accs, pred_dict


# ssl._create_default_https_context = ssl._create_unverified_context
#
# base_model, _ = initialize_model('resnet', 5, False)
#
# model = AgnosticModel(base_model)
#
# for i in range(1, 5):
#     test_accs, model = training_loop_w_pretrain('batched_data/b%i.npy' % i, 'batched_data/labels.npy', 10, model)
#
# filehandler = open('resnet_trained.obj', 'wb')
# pickle.dump(model, filehandler)
