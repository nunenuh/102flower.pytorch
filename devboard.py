{
  "cells": [
    {
      "metadata": {
        "_kg_hide-output": true,
        "_uuid": "82dba2dc7059a766fe05ddb328c6acb60533dd8f"
      },
      "cell_type": "markdown",
      "source": "# Package Imports\nAll the necessary packages and modules are imported in the first cell of the notebook"
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "826bbf3fd78d1e4ffe325909d3c80f7aff88f314"
      },
      "cell_type": "code",
      "source": "# Imports here\n%matplotlib inline\nimport numpy as np\nimport cv2\nimport os\nimport sys\nimport time\nimport json\nimport shutil\n\nimport PIL\nimport PIL.Image\n\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport torch.optim as optim\nfrom torch.utils import data\nfrom torch.autograd import Variable\nimport torch.nn.parallel\nimport torch.backends.cudnn as cudnn\nimport torch.optim as optim\n\nimport torchvision\nimport torchvision.datasets as dset\nimport torchvision.transforms as transforms\nimport torchvision.utils as vutils\n\nimport matplotlib.pyplot as plt\nfrom matplotlib import gridspec\nimport matplotlib.pyplot as plt\nimport matplotlib.animation as animation\nfrom IPython.display import HTML",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "826bbf3fd78d1e4ffe325909d3c80f7aff88f314"
      },
      "cell_type": "markdown",
      "source": "# Defining Helper"
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "826bbf3fd78d1e4ffe325909d3c80f7aff88f314"
      },
      "cell_type": "code",
      "source": "class NormalizeInverse(torchvision.transforms.Normalize):\n    \"\"\"\n    Undoes the normalization and returns the reconstructed images in the input domain.\n    \"\"\"\n\n    def __init__(self, mean, std):\n        mean = torch.as_tensor(mean)\n        std = torch.as_tensor(std)\n        std_inv = 1 / (std + 1e-7)\n        mean_inv = -mean * std_inv\n        super().__init__(mean=mean_inv, std=std_inv)\n\n    def __call__(self, tensor):\n        return super().__call__(tensor.clone())",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "826bbf3fd78d1e4ffe325909d3c80f7aff88f314"
      },
      "cell_type": "markdown",
      "source": "# Version Check"
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "826bbf3fd78d1e4ffe325909d3c80f7aff88f314"
      },
      "cell_type": "code",
      "source": "print('Torch Version\\t',torch.__version__)\nprint('PIL Version\\t',PIL.__version__)\nprint('Torchvision\\t',torchvision.__version__)\nprint('GPU Available \\t',torch.cuda.is_available())",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "11b4b040542136e41b18be6488c3e0564c35b9a8"
      },
      "cell_type": "markdown",
      "source": "# HyperParameter Set"
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "11b4b040542136e41b18be6488c3e0564c35b9a8"
      },
      "cell_type": "code",
      "source": "NGPU = 1\nNUM_EPOCH = 100\nBSIZE = 32\nLRATE = 0.005\nMOMENTUM=0.9\n\nLRSTEP=7\nGAMMA=0.1\nPRINT_FREQ = 5",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "ad8fe96c8df32ff9a6d8482b767a5f247f14bae7"
      },
      "cell_type": "code",
      "source": "device = torch.device(\"cuda:0\" if (torch.cuda.is_available() and NGPU > 0) else \"cpu\")\nprint(device)",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "723e5feded0b289f0643b48e8b05145dbdde74d5"
      },
      "cell_type": "markdown",
      "source": "# Data Preparation\n## Training and Validatation dataset\n\n### Augmentation, Data normalization and Dataset Preparation\n\n* torchvision transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping\n* The training, validation, and testing data is appropriately cropped and normalized\n\n"
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "723e5feded0b289f0643b48e8b05145dbdde74d5"
      },
      "cell_type": "code",
      "source": "data_dir = '/kaggle/input/flower_data/flower_data'\ntrain_dir = data_dir + '/train'\nvalid_dir = data_dir + '/valid'",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "d128928ce1fde888b03da4af782de3f006f81927"
      },
      "cell_type": "code",
      "source": "# TODO: Define your transforms for the training and validation sets\n#  scaling, rotations, mirroring, and/or cropping\nmean_val, std_val = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]\ntrain_transforms =  transforms.Compose([\n    transforms.RandomRotation(30),\n    transforms.RandomResizedCrop(224),\n    transforms.RandomHorizontalFlip(),\n    transforms.ToTensor(),\n    transforms.Normalize(mean_val,std_val)\n])\n\nvalid_transforms =  transforms.Compose([\n    transforms.Resize(255),\n    transforms.CenterCrop(224),\n    transforms.ToTensor(),\n    transforms.Normalize(mean_val,std_val)\n])\n\n# TODO: Load the datasets with ImageFolder\ntrain_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)\nvalid_dataset = torchvision.datasets.ImageFolder(valid_dir, transform=valid_transforms)\n\nCLAZZ = len(train_dataset.classes)",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "ea8bd91ea8ae46a9456d3fa45cc0e03a9e5a0d53"
      },
      "cell_type": "markdown",
      "source": "# Data batching and Data loading\n\n* The data for each set is loaded with torchvision's DataLoader\n* The data for each set (train, validation, test) is loaded with torchvision's ImageFolder"
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "ea8bd91ea8ae46a9456d3fa45cc0e03a9e5a0d53"
      },
      "cell_type": "code",
      "source": "train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BSIZE, shuffle=True, num_workers=0)\nvalid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BSIZE, shuffle=True, num_workers=0)",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "475667709c393193a6b4f298f43de079408b10ae"
      },
      "cell_type": "code",
      "source": "# get category to flower name\ndef get_all_flower_names():\n    with open('/kaggle/input/cat_to_name.json', 'r') as f:\n            cat_to_name = json.load(f)\n    return cat_to_name\n\ndef flower_name(val, array_index=False):\n    labels = get_all_flower_names()\n    if array_index:\n        val = val + 1\n    return labels[str(val)]\n\nFLOWER_LABELS = get_all_flower_names()\n#test function\n# print(FLOWER_LABELS)",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "7c83986f9bfba464e82e62292b0a692c48f53feb"
      },
      "cell_type": "code",
      "source": "# obtain one batch of training images\ndenorm = NormalizeInverse(mean_val, std_val)\ndataiter = iter(train_loader)\nimages, labels = dataiter.next()",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "7c83986f9bfba464e82e62292b0a692c48f53feb"
      },
      "cell_type": "code",
      "source": "# plot the images in the batch, along with the corresponding labels\nfig = plt.figure(figsize=(25, 4))\n# display 20 images\nfor idx in np.arange(BSIZE-1):\n    ax = fig.add_subplot(2, BSIZE/2, idx+1, xticks=[], yticks=[])\n    img = denorm(images[idx]).permute(1,2,0).numpy()\n    img = np.clip(img, 0, 1)\n    plt.imshow(img, cmap='gray')\n    ax.set_title(flower_name(labels[idx].item(), array_index=True))",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "_uuid": "81d2a3a2610a2bac6be8f6afd345afe4da7c8267"
      },
      "cell_type": "markdown",
      "source": "# Building and training the classifier\n## Pretrained Network and Feedforward Classifier\n\n* A pretrained network such as VGG16 is loaded from torchvision.models and the parameters are frozen\n* A new feedforward network is defined for use as a classifier using the features as input\n"
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "d6eb396c9b76cfb8db0c7beab3f48add24904da6",
        "scrolled": true
      },
      "cell_type": "code",
      "source": "class AdaptiveConcatPool2d(nn.Module):\n    \"Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`.\"\n    def __init__(self, sz:int=None):\n        \"Output will be 2*sz or 2 if sz is None\"\n        super().__init__()\n        sz = sz or 1\n        self.ap,self.mp = nn.AdaptiveAvgPool2d(sz), nn.AdaptiveMaxPool2d(sz)\n    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)\n\nclass Lambda(nn.Module):\n    \"An easy way to create a pytorch layer for a simple `func`.\"\n    def __init__(self, func):\n        \"create a layer that simply calls `func` with `x`\"\n        super().__init__()\n        self.func=func\n\n    def forward(self, x): return self.func(x)\n    \ndef Flatten()->torch.Tensor:\n    \"Flattens `x` to a single dimension, often used at the end of a model.\"\n    return Lambda(lambda x: x.view((x.size(0), -1)))\n    \n\n# TODO: Build and train your network\nclass ResidualFlowerNetwork(nn.Module):\n    def __init__(self, resnet, clazz):\n        super(ResidualFlowerNetwork, self).__init__()\n        self.pool_size = 1\n        self.resnet = resnet\n        \n        # out_channels multiple by pool size and multiply by 2\n        # multiply by 2 is get from torch cat of AdaptiveAvgPool2d and AdaptiveMaxPool2d\n        in_features = self.get_last_layer_out_channels() * self.pool_size*self.pool_size*2\n        \n        self.resnet.avgpool = nn.Sequential(\n            AdaptiveConcatPool2d(self.pool_size),\n            Flatten()\n        )\n        \n        self.resnet.fc = nn.Sequential(\n            nn.BatchNorm1d(in_features),\n            nn.Dropout(0.5),\n            nn.Linear(in_features, in_features//2),\n            nn.ReLU(inplace=True),\n            \n            nn.BatchNorm1d(in_features//2),\n            nn.Dropout(0.3),\n            nn.Linear(in_features//2, in_features//4),\n            nn.ReLU(inplace=True),\n            \n            nn.BatchNorm1d(in_features//4),\n            nn.Dropout(0.2),\n            nn.Linear(in_features//4, clazz),\n#             nn.ReLU(inplace=True)\n#             nn.Linear(512, clazz)\n        )\n    def get_last_layer_out_channels(self):\n        if type(self.resnet.layer4[2]) == torchvision.models.resnet.BasicBlock:\n            return self.resnet.layer4[2].conv2.out_channels\n        elif type(self.resnet.layer4[2]) == torchvision.models.resnet.Bottleneck:\n            return self.resnet.layer4[2].conv3.out_channels\n        else:\n            return 0\n        \n    def freeze(self):\n        for param in self.resnet.parameters():\n            param.require_grad = False\n        for param in self.resnet.fc.parameters():\n            param.require_grad= True\n            \n    def unfreeze(self):\n        for param in self.resnet.parameters():\n            param.require_grad = True\n    \n    def forward(self, x):\n        x = self.resnet(x)\n        return x\n\n    \nresnet = torchvision.models.resnet34(pretrained=True)\n# print(resnet.fc.in_features) \nmodel = ResidualFlowerNetwork(resnet, CLAZZ)\n\nif torch.cuda.is_available():\n    model.cuda()\nmodel    \n# criterion = nn.CrossEntropyLoss()\n# optimizer = optim.SGD(model.resnet.fc.parameters(), lr=LRATE, momentum=MOMENTUM)\n# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LRSTEP, gamma=GAMMA)",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "d6eb396c9b76cfb8db0c7beab3f48add24904da6"
      },
      "cell_type": "code",
      "source": "# x = torch.rand(2,3,224,224).cuda()\n# model(x)",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "7d026ec582f9d1d90f437d11e5b9b2dcdc0eb389"
      },
      "cell_type": "markdown",
      "source": "# Training the network\nThe parameters of the feedforward classifier are appropriately trained, while the parameters of the feature network are left static"
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "7d026ec582f9d1d90f437d11e5b9b2dcdc0eb389"
      },
      "cell_type": "code",
      "source": "batch_history = {\n    'train':{'epoch':[], 'loss':[], 'acc_topk1':[], 'acc_topk5':[]},\n    'valid':{'epoch':[], 'loss':[], 'acc_topk1':[], 'acc_topk5':[]}\n}\n\ndef train(train_loader, model, criterion, optimizer, epoch, print_freq, save_history=True, ngpu=1):\n    device = torch.device(\"cuda:0\" if (torch.cuda.is_available() and ngpu > 0) else \"cpu\")\n    batch_time = AverageMeter()\n    data_time = AverageMeter()\n    losses = AverageMeter()\n    top1 = AverageMeter()\n    top5 = AverageMeter()\n    history = {'epoch':[], 'loss':[], 'acc_topk1':[], 'acc_topk5':[]}\n\n    # switch to train mode\n    model.train()\n\n    end = time.time()\n    for i, (input, target) in enumerate(train_loader):\n        # measure data loading time\n        data_time.update(time.time() - end)\n\n        input = input.to(device)\n        target = target.to(device)\n        \n        optimizer.zero_grad()\n        \n        # compute output\n        output = model(input)\n        loss = criterion(output, target)\n\n        # measure accuracy and record loss\n        acc1, acc5 = accuracy(output, target, topk=(1, 5))\n        losses.update(loss.item(), input.size(0))\n        top1.update(acc1[0], input.size(0))\n        top5.update(acc5[0], input.size(0))\n\n        # compute gradient and do SGD step\n        \n        loss.backward()\n        optimizer.step()\n\n        # measure elapsed time\n        batch_time.update(time.time() - end)\n        end = time.time()\n\n        if i % print_freq == 0:\n            history['epoch'].append(epoch)\n            history['loss'].append(losses.avg)\n            history['acc_topk1'].append(top1.avg)\n            history['acc_topk5'].append(top5.avg)\n            print('Epoch: [{0}][{1}/{2}]\\t'\n                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\\t'\n                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\\t'\n                  'Loss {loss.val:.4f} ({loss.avg:.4f})\\t'\n                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\\t'\n                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(\n                   epoch, i, len(train_loader), batch_time=batch_time,\n                   data_time=data_time, loss=losses, top1=top1, top5=top5))\n    return history\n\n\ndef validate(val_loader, model, criterion, epoch, print_freq, save_history=True, ngpu=1):\n    device = torch.device(\"cuda:0\" if (torch.cuda.is_available() and ngpu > 0) else \"cpu\")\n    batch_time = AverageMeter()\n    losses = AverageMeter()\n    top1 = AverageMeter()\n    top5 = AverageMeter()\n    history = {'epoch':[], 'loss':[], 'acc_topk1':[], 'acc_topk5':[]}\n\n    # switch to evaluate mode\n    model.eval()\n\n    with torch.no_grad():\n        end = time.time()\n        for i, (input, target) in enumerate(val_loader):\n            \n            input = input.to(device)\n            target = target.to(device)\n\n            # compute output\n            output = model(input)\n            loss = criterion(output, target)\n\n            # measure accuracy and record loss\n            acc1, acc5 = accuracy(output, target, topk=(1, 5))\n            losses.update(loss.item(), input.size(0))\n            top1.update(acc1[0], input.size(0))\n            top5.update(acc5[0], input.size(0))\n\n            # measure elapsed time\n            batch_time.update(time.time() - end)\n            end = time.time()\n\n            if i % print_freq == 0:\n                history['epoch'].append(epoch)\n                history['loss'].append(losses.avg)\n                history['acc_topk1'].append(top1.avg)\n                history['acc_topk5'].append(top5.avg)\n                print('Test: [{0}/{1}]\\t'\n                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\\t'\n                      'Loss {loss.val:.4f} ({loss.avg:.4f})\\t'\n                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\\t'\n                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(\n                       i, len(val_loader), batch_time=batch_time, loss=losses,\n                       top1=top1, top5=top5))\n\n        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'\n              .format(top1=top1, top5=top5))\n    \n    return history\n\n\ndef save_checkpoint(state, is_best, filename='checkpoint.pth'):\n    torch.save(state, filename)\n    if is_best:\n        shutil.copyfile(filename, 'model_best.pth')\n\n\nclass AverageMeter(object):\n    \"\"\"Computes and stores the average and current value\"\"\"\n    def __init__(self):\n        self.reset()\n\n    def reset(self):\n        self.val = 0\n        self.avg = 0\n        self.sum = 0\n        self.count = 0\n\n    def update(self, val, n=1):\n        self.val = val\n        self.sum += val * n\n        self.count += n\n        self.avg = self.sum / self.count\n\n\ndef adjust_learning_rate(optimizer, epoch, decay, lrate):\n    \"\"\"Sets the learning rate to the initial LR decayed by 10 every 30 epochs\"\"\"\n    lr = lrate * (0.1 ** (epoch // decay))\n    for param_group in optimizer.param_groups:\n        param_group['lr'] = lr\n\n\ndef accuracy(output, target, topk=(1,)):\n    \"\"\"Computes the accuracy over the k top predictions for the specified values of k\"\"\"\n    with torch.no_grad():\n        maxk = max(topk)\n        batch_size = target.size(0)\n        \n        _, pred = output.topk(maxk, 1, True, True)\n        pred = pred.t()\n        correct = pred.eq(target.view(1, -1).expand_as(pred))\n\n        res = []\n        for k in topk:\n            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)\n            res.append(correct_k.mul_(100.0 / batch_size))\n        return res\n       \ndef getsteplr(base_lr=0.001, max_lr=0.1, step=4):\n    lr = base_lr\n    hlr = max_lr\n    step = hlr/(step-1)\n    step_lr = np.arange(lr, hlr+step, step).tolist()\n    return step_lr\ngetsteplr(base_lr=0.001, step=6)\n    ",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "7d026ec582f9d1d90f437d11e5b9b2dcdc0eb389",
        "scrolled": true
      },
      "cell_type": "code",
      "source": "best_acc1 = 0\nhistory = {'epoch':[], 'train_detail':[],'valid_detail':[],}\n\nNUM_EPOCH=34\n#train only classifier\nmodel.freeze()\nmodel = model.cuda()\n\nLRATE=0.05\ncriterion = nn.CrossEntropyLoss()\noptimizer = optim.SGD(model.resnet.fc.parameters(), lr=LRATE, momentum=MOMENTUM)\nscheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LRSTEP, gamma=GAMMA)\n\nfor epoch in range(NUM_EPOCH): \n    prev_time = time.time()\n    scheduler.step()\n    # train pretrained network\n    if epoch == 13:\n        model.unfreeze()\n        step_lr = getsteplr(base_lr=LRATE/100, max_lr=LRATE, step=6)\n        print(step_lr)\n        optimizer = optim.SGD(\n            [\n                {'params': model.resnet.conv1.parameters()},\n                {'params': model.resnet.bn1.parameters()},\n                {'params': model.resnet.relu.parameters()},\n                {'params': model.resnet.maxpool.parameters()},\n                {'params': model.resnet.layer1.parameters(), 'lr':step_lr[1]},\n                {'params': model.resnet.layer2.parameters(), 'lr':step_lr[2]},\n                {'params': model.resnet.layer3.parameters(), 'lr':step_lr[3]},\n                {'params': model.resnet.layer4.parameters(), 'lr':step_lr[4]},\n                {'params': model.resnet.avgpool.parameters(), 'lr':step_lr[4]},\n                {'params': model.resnet.fc.parameters(), 'lr': step_lr[4]}\n            ],\n            lr=step_lr[0])\n#         scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LRSTEP, gamma=GAMMA)\n    \n    # train for one epoch\n    train_history = train(train_loader, model, criterion, optimizer, epoch, print_freq=PRINT_FREQ)\n    # evaluate on validation set\n    valid_history = validate(valid_loader, model, criterion, epoch, print_freq=PRINT_FREQ)\n    acc1 = valid_history['acc_topk1'][len(valid_history['acc_topk1'])-1]\n    \n    history['epoch'].append(epoch)\n    history['train_detail'].append(train_history)\n    history['valid_detail'].append(valid_history)\n    \n    # remember best acc@1 and save checkpoint\n    is_best = acc1 > best_acc1\n    best_acc1 = max(acc1, best_acc1)\n    save_checkpoint({\n        'epoch': epoch + 1,\n        'batch_size': BSIZE,\n        'learning_rate': LRATE,\n        'total_clazz': CLAZZ,\n        'class_to_idx': train_dataset.class_to_idx,\n        'labels': FLOWER_LABELS,\n        'history': history,\n        'arch': 'resnet101',\n        'state_dict': model.state_dict(),\n        'best_acc_topk1': best_acc1,\n        'optimiz1er' : optimizer.state_dict(),\n    }, is_best)\n    \n    curr_time = time.time()\n    total_time = curr_time - prev_time\n    print(f'Total Time / Epcoh : {total.time}')",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "_uuid": "6d028cb246ecb32a007dcf6028e472fc15608551"
      },
      "cell_type": "markdown",
      "source": "## Save the checkpoint\n\nNow that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.\n\n```model.class_to_idx = image_datasets['train'].class_to_idx```\n\nRemember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now."
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "aad43061dbf5442d5aab37d52fd99b36a608ae38",
        "scrolled": false
      },
      "cell_type": "code",
      "source": "# TODO: Save the checkpoint\n# train_dataset.class_to_idx\n# checkpoint = torch.load('checkpoint.pth.tar')\n# checkpoint['history']['train']",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "_uuid": "6b5c8d3c8ab058999869c406b1c8399714bdaeb9"
      },
      "cell_type": "markdown",
      "source": "## Loading the checkpoint\n\nAt this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network."
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "87943cf3b896a6c6af277502b675b6ac4a2eda3b"
      },
      "cell_type": "code",
      "source": "# TODO: Write a function that loads a checkpoint and rebuilds the model\ndef load_flower_network(filename):\n    if os.path.isfile(filename): \n        checkpoint = torch.load(filename)\n        resnet = torchvision.models.resnet34(pretrained=True)\n        clazz = checkpoint['total_clazz']\n        model = ResidualFlowerNetwork(resnet, clazz)\n        model.load_state_dict(checkpoint['state_dict'])\n        return model\n    else:\n        return None\n    \n\ndef load_checkpoint(filename):\n    if os.path.isfile(filename): \n        checkpoint = torch.load(filename)\n        return checkpoint\n    else:\n        return None\n    \nmodel = load_flower_network('checkpoint.pth')\ncheckpoint = load_checkpoint('checkpoint.pth')",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "87943cf3b896a6c6af277502b675b6ac4a2eda3b"
      },
      "cell_type": "code",
      "source": "model",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "_uuid": "c6505f32a43a57611ec6e14d1e71cbc45de28eb3"
      },
      "cell_type": "markdown",
      "source": "# Inference for classification\n\n## Image Preprocessing"
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "f096375811ffdaf812a87bdf6320f4bab54a1520"
      },
      "cell_type": "code",
      "source": "def process_image(image):\n    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,\n        returns an Numpy array\n    '''\n    im = PIL.Image.open(image)\n    mean_val, std_val = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]\n    do_transforms =  transforms.Compose([\n        transforms.Resize(255),\n        transforms.CenterCrop(224),\n        transforms.ToTensor(),\n        transforms.Normalize(mean_val,std_val)\n    ])\n    im_tfmt = do_transforms(im)\n    im_add_batch = im_tfmt.view(1, im_tfmt.shape[0], im_tfmt.shape[1], im_tfmt.shape[2])\n    return im_add_batch",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "f096375811ffdaf812a87bdf6320f4bab54a1520"
      },
      "cell_type": "code",
      "source": "# test process image\nvalid_dir = data_dir + '/valid/'\nimage_file = valid_dir +\"10/image_07094.jpg\"\nout_im = process_image(image_file)\nout_im.shape",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "_uuid": "b8a394de3a61d424a4c30c163a6051b4a830c08f"
      },
      "cell_type": "markdown",
      "source": "To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions)."
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "37e04a6f9ede81d74ba93d7fe89ee2e1464e176e"
      },
      "cell_type": "code",
      "source": "def imshow(image, ax=None, title=None):\n    \"\"\"Imshow for Tensor.\"\"\"\n    if ax is None:\n        fig, ax = plt.subplots()\n    \n    # PyTorch tensors assume the color channel is the first dimension\n    # but matplotlib assumes is the third dimension\n    image = image.numpy().transpose((1, 2, 0))\n    \n    # Undo preprocessing\n    mean = np.array([0.485, 0.456, 0.406])\n    std = np.array([0.229, 0.224, 0.225])\n    image = std * image + mean\n    \n    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed\n    image = np.clip(image, 0, 1)\n    \n    ax.imshow(image)\n    \n    return ax\n\nimshow(out_im.squeeze())",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "_uuid": "2ea31efe59ad0c4b14c4dbe7ffb738afb52084a1"
      },
      "cell_type": "markdown",
      "source": "## Class Prediction\n\nOnce you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.\n\nTo get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.\n\nAgain, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.\n\n```python\nprobs, classes = predict(image_path, model)\nprint(probs)\nprint(classes)\n> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]\n> ['70', '3', '45', '62', '55']\n```"
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "af1292aff4f4b277b77e2dcb4356cf2f66d8c3b1"
      },
      "cell_type": "code",
      "source": "def getFlowerClassIndex(classes, class_to_idx):\n    idx_to_class = {val: key for key, val in class_to_idx.items()}\n    class_to_flower_class_idx = [idx_to_class[lab] for lab in classes.squeeze().numpy().tolist()]\n    flower_class_to_name = [flower_name(cls_idx) for cls_idx in class_to_flower_class_idx]\n    return class_to_flower_class_idx, flower_class_to_name\n\ndef predict(image_path, model, topk=5):\n    ''' Predict the class (or classes) of an image using a trained deep learning model.\n    '''\n    image = process_image(image_path)\n    model.eval()\n    model = model.cpu()\n    with torch.no_grad():\n        output = model.forward(image)\n        output = F.log_softmax(output, dim=1)\n        ps = torch.exp(output)\n        result = ps.topk(topk, dim=1, largest=True, sorted=True)\n        \n    return result\n\n\nvalid_dir = data_dir + '/valid/'\nimage_file = valid_dir +\"10/image_07094.jpg\"\nprobs, classes = predict(image_file, model, topk=5)\nclass_index, class_name = getFlowerClassIndex(classes, checkpoint['class_to_idx'])\n\nprint(probs)\nprint(classes)\nprint(class_index)\nprint(class_name)",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "_uuid": "4b30db33a81fea5c446199ca6ca9c4e0b304a083"
      },
      "cell_type": "markdown",
      "source": "## Sanity Checking\n\nNow that you can use a trained model for predictions, check to make sure it makes sense. Even if the validation accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:\n\n<img src='assets/inference_example.png' width=300px>\n\nYou can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above."
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "db69479ee312a17e44e95d836df2ec3537005995"
      },
      "cell_type": "code",
      "source": "def view_classify(img_path, label_idx, prob, classes, class_to_idx):\n    ''' Function for viewing an image and it's predicted classes.\n    '''\n    img = np.asarray(PIL.Image.open(img_path))\n    ps = prob.data.numpy().squeeze().tolist()\n    fig, (ax1, ax2) = plt.subplots(figsize=(10,10), ncols=2)\n    ax1.imshow(img.squeeze())\n    ax1.set_title(flower_name(label_idx))\n    ax1.axis('off')\n    \n    ax2.barh(np.arange(5), ps)\n    ax2.set_aspect(0.2)\n    ax2.set_yticks(np.arange(5))\n    \n    \n    class_idx, class_name = getFlowerClassIndex(classes, class_to_idx)\n    ax2.set_yticklabels(class_name, size='large');\n    ax2.set_title('Class Probability')\n    ax2.set_xlim(0, 1.1)\n\n    plt.tight_layout()\n\n\nvalid_dir = data_dir + '/valid/'\nimage_file = valid_dir +\"2/image_05136.jpg\"\nprobs, classes = predict(image_file, model)\nprint(probs)\nview_classify(image_file, 2, probs, classes, checkpoint['class_to_idx'])",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "b8f32ce0ba3b013667bc9c774111abfbba055e05"
      },
      "cell_type": "code",
      "source": "# track test loss\n\nvalid_loader2 = torch.utils.data.DataLoader(valid_dataset, batch_size=BSIZE, shuffle=True, num_workers=0)\ntest_loss = 0.0\nclass_correct = list(0. for i in range(102))\nclass_total = list(0. for i in range(102))\nbatch_size = BSIZE\nmodel.eval()\nmodel.to(device)\n# iterate over test data\nfor data, target in valid_loader2:\n    # move tensors to GPU if CUDA is available\n    data, target = data.to(device), target.to(device)\n    # forward pass: compute predicted outputs by passing inputs to the model\n    output = model(data)\n    # calculate the batch loss\n    loss = criterion(output, target)\n    # update test loss \n    test_loss += loss.item()*data.size(0)\n    # convert output probabilities to predicted class\n    _, pred = torch.max(output, 1)    \n    # compare predictions to true label\n    correct_tensor = pred.eq(target.data.view_as(pred))\n    correct = np.squeeze(correct_tensor.numpy())\n    # calculate test accuracy for each object class\n    batch_size = target.size(0)\n    for i in range(batch_size):\n        label = target.data[i]\n        class_correct[label] += correct[i].item()\n        class_total[label] += 1",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "b8f32ce0ba3b013667bc9c774111abfbba055e05"
      },
      "cell_type": "code",
      "source": "# average test loss\ntest_loss = test_loss/len(valid_loader.dataset)\nprint('Test Loss: {:.6f}\\n'.format(test_loss))\n# valid_loader.dataset.class_to_idx['1'], class_correct[0], class_total[0]\nprint(getFlowerClassIndex(pred.cpu(), valid_loader.dataset.class_to_idx))\nfor i in range():\n    if class_total[i] > 0:\n        total = 100 * class_correct[i] / class_total[i]\n        total_correct = np.sum(class_correct[i])\n        total_class = np.sum(class_total[i])\n        clzz = valid_loader.dataset.class_to_idx[str(i+1)]\n        print(f'Test Accuracy of {clzz}: {total}% ({total_correct}/{total_class})')\n        \n    else:\n        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))\n\nprint('\\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (\n    100. * np.sum(class_correct) / np.sum(class_total),\n    np.sum(class_correct), np.sum(class_total)))",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "48878f7bf3ef862781c820e763de6287229b2aa4"
      },
      "cell_type": "code",
      "source": "# valid_history = validate(valid_loader, model, criterion, epoch, print_freq=PRINT_FREQ)\n",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "19491dcb5d7dbad5cfb594fd5f5cfd441c6499ce"
      },
      "cell_type": "code",
      "source": "# # valid_history\n!ls -al",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "bf54879f4c41d563ae5e6cb339ce0d108111d5c0"
      },
      "cell_type": "code",
      "source": "# !mv checkpoint.pth resnet_101_checkpoint.pth",
      "execution_count": null,
      "outputs": []
    },
    {
      "metadata": {
        "trusted": true,
        "_uuid": "969739d66c3da00d62ffe577c999fdee8b9d2e75"
      },
      "cell_type": "code",
      "source": "# !mv checkpoint.pth resnet_101_checkpoint.pth",
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.6.6",
      "mimetype": "text/x-python",
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "pygments_lexer": "ipython3",
      "nbconvert_exporter": "python",
      "file_extension": ".py"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 1
}