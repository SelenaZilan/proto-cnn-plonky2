from __future__ import print_function
from __future__ import division
import argparse
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import copy
import pathlib
from models import SmallIrisCNN
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data', metavar='DIR',default="./data_256_stacked",
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='smalliris',
                    help='model architecture (default: smalliris)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-n_classes',default=1500, type=int, help='number of classses')
parser.add_argument('--input-size', default=None, type=int, help='override student input size')
parser.add_argument('--embedding-dim', default=128, type=int, help='SmallIris embedding dimension / final conv channels')
parser.add_argument('--smalliris-c1', default=32, type=int, help='SmallIris first conv channels')
parser.add_argument('--smalliris-c2', default=64, type=int, help='SmallIris second conv channels')
parser.add_argument('--teacher-arch', default=None, choices=['resnet101'], help='optional teacher for distillation')
parser.add_argument('--teacher-checkpoint', default=None, help='teacher checkpoint path')
parser.add_argument('--distill-alpha', default=0.3, type=float, help='weight on KL distillation loss')
parser.add_argument('--distill-temperature', default=4.0, type=float, help='softmax temperature for KD')
parser.add_argument('--output-dir', default='.', help='directory to save best checkpoint')
parser.add_argument('--train-aug', default='center_crop', choices=['center_crop', 'random_resized_crop'],
                    help='training augmentation for ImageFolder pipeline')
parser.add_argument('--scheduler', default='cosine', choices=['none', 'cosine'],
                    help='learning-rate scheduler')
parser.add_argument('--min-lr', default=1e-5, type=float, help='minimum learning rate for cosine scheduler')
parser.add_argument('--label-smoothing', default=0.1, type=float, help='label smoothing for CE loss')
def get_dataloaders(data_dir, input_size, batch_size, train_aug='center_crop'):
    print("Initializing Datasets and Dataloaders...")

    if train_aug == 'random_resized_crop':
        train_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomResizedCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    data_transforms = {
        'train': train_transform,
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    return {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
        ['train', 'val']}
def initialize_model(
    num_classes,
    learning_rate,
    input_size_override=None,
    embedding_dim=128,
    smalliris_c1=32,
    smalliris_c2=64,
    weight_decay=1e-4,
    label_smoothing=0.0,
):
    model = SmallIrisCNN(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        c1=smalliris_c1,
        c2=smalliris_c2,
    )
    input_size = 64 if input_size_override is None else input_size_override
    params_to_update = model.parameters()
    optimizer = optim.AdamW(params_to_update, lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    return model, optimizer, criterion, input_size
def load_teacher_model(teacher_arch, teacher_checkpoint, num_classes, device):
    if teacher_arch is None:
        return None, None
    if teacher_arch != "resnet101":
        raise ValueError(f"Unsupported teacher_arch: {teacher_arch}")
    if teacher_checkpoint is None:
        raise ValueError("teacher_checkpoint is required when teacher_arch is set")

    teacher = models.resnet101(pretrained=False)
    num_ftrs = teacher.fc.in_features
    teacher.fc = nn.Linear(num_ftrs, num_classes)

    payload = torch.load(teacher_checkpoint, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    teacher.load_state_dict(payload)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher, 224
def resize_for_student(inputs, student_input_size):
    if inputs.shape[-1] == student_input_size and inputs.shape[-2] == student_input_size:
        return inputs
    return F.interpolate(inputs, size=(student_input_size, student_input_size), mode='bilinear', align_corners=False)
def distillation_loss(student_logits, teacher_logits, labels, criterion, alpha, temperature):
    ce = criterion(student_logits, labels)
    kd = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
    ) * (temperature ** 2)
    loss = (1.0 - alpha) * ce + alpha * kd
    return loss, ce, kd
def save_checkpoint(model_name, model, num_epochs, learning_rate, output_dir, input_size):
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{model_name}_e_{num_epochs}_lr_{str(learning_rate).replace('.','_')}"
    if hasattr(model, "config_dict"):
        cfg = model.config_dict()
        filename += f"_in_{input_size}_c1_{cfg.get('c1')}_c2_{cfg.get('c2')}_emb_{cfg.get('embedding_dim')}"
    filename += "_best.pth"
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            **(model.config_dict() if hasattr(model, "config_dict") else {"num_classes": model.fc.out_features}),
            "input_size": input_size,
        },
    }
    torch.save(payload, out_dir / filename)
    return out_dir / filename
def train_model(
    model_name,
    model,
    dataloaders,
    criterion,
    optimizer,
    device,
    learning_rate,
    num_epochs=25,
    student_input_size=64,
    teacher_model=None,
    teacher_input_size=None,
    distill_alpha=0.3,
    distill_temperature=4.0,
    output_dir='.',
    scheduler_name='none',
    min_lr=1e-5,
):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    epoch_runtimes_history = []
    scheduler = None
    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

    for epoch in range(num_epochs):
        if epoch != 0:
            endtime_estimation = (num_epochs - epoch) * np.mean(epoch_runtimes_history)
        print(f'Epoch {epoch}/{num_epochs - 1} - estimated time left: {"starting..." if len(epoch_runtimes_history) == 0 else str(datetime.timedelta(seconds=endtime_estimation))}')
        print('-' * 10)

        epoch_start_time = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                teacher_inputs = inputs.to(device)
                student_inputs = resize_for_student(teacher_inputs, student_input_size)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(student_inputs)
                    if phase == 'train' and teacher_model is not None:
                        with torch.no_grad():
                            if teacher_input_size is not None and (
                                teacher_inputs.shape[-1] != teacher_input_size
                                or teacher_inputs.shape[-2] != teacher_input_size
                            ):
                                teacher_inputs_kd = resize_for_student(teacher_inputs, teacher_input_size)
                            else:
                                teacher_inputs_kd = teacher_inputs
                            teacher_logits = teacher_model(teacher_inputs_kd)
                        loss, ce_loss, kd_loss = distillation_loss(
                            outputs,
                            teacher_logits,
                            labels,
                            criterion,
                            distill_alpha,
                            distill_temperature,
                        )
                    else:
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == "val":
                print('best val acc: {:4f}'.format(best_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                ckpt_path = save_checkpoint(
                    model_name,
                    model,
                    num_epochs,
                    learning_rate,
                    output_dir,
                    student_input_size,
                )
                print(f"Saved best checkpoint to {ckpt_path}")

            if phase == 'val':
                val_acc_history.append(epoch_acc)

        if scheduler is not None:
            scheduler.step()
            print(f"lr now: {scheduler.get_last_lr()[0]:.6g}")

        epoch_runtime = time.time() - epoch_start_time
        epoch_runtimes_history.append(epoch_runtime)
        print(f"Epoch runtime: {str(datetime.timedelta(seconds=epoch_runtime))}")
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history

if __name__ == '__main__':
    args = parser.parse_args()

    data_dir = args.data
    model_name = args.arch
    num_classes = args.n_classes
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.lr

    student_input_size = args.input_size
    model, optimizer, criterion, input_size = initialize_model(
        num_classes,
        learning_rate,
        input_size_override=student_input_size,
        embedding_dim=args.embedding_dim,
        smalliris_c1=args.smalliris_c1,
        smalliris_c2=args.smalliris_c2,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
    )

    print(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    teacher_model, teacher_input_size = load_teacher_model(
        args.teacher_arch,
        args.teacher_checkpoint,
        num_classes,
        device,
    )
    dataloader_input_size = max(input_size, teacher_input_size or input_size)

    dataloaders_dict = get_dataloaders(data_dir, dataloader_input_size, batch_size, train_aug=args.train_aug)

    model = model.to(device)

    model_ft, hist = train_model(
        model_name,
        model,
        dataloaders_dict,
        criterion,
        optimizer,
        device,
        learning_rate,
        num_epochs=num_epochs,
        student_input_size=input_size,
        teacher_model=teacher_model,
        teacher_input_size=teacher_input_size,
        distill_alpha=args.distill_alpha,
        distill_temperature=args.distill_temperature,
        output_dir=args.output_dir,
        scheduler_name=args.scheduler,
        min_lr=args.min_lr,
    )
