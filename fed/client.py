import os
os.environ["GRPC_ARG_KEEPALIVE_TIME_MS"] = "60000"
os.environ["GRPC_ARG_KEEPALIVE_TIMEOUT_MS"] = "5000"
os.environ["GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS"] = "1"
# CUDA_VISIBLE_DEVICES=0 nohup python client.py 0 > client_0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python client.py 1 > client_1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python client.py 2 > client_2.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python server.py > server.log 2>&1 &
# pkill -f server.py
# pkill -f client.py

import sys
import datetime
import torch
import flwr as fl
from collections import OrderedDict
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import Loss, get_lr_scheduler, set_optimizer_lr
from nets.distribution_generator import DistributionGenerator
from nets.discriminator import FeatureDiscriminator

from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_classes, seed_everything
from utils.utils_fit import fit_one_epoch
from utils.callbacks import LossHistory, EvalCallback

seed_everything(11)

client_id = int(sys.argv[1])

classes_path = 'model_data/gpr_classes.txt'
train_path = f'./client_data/client_{client_id}_train.txt'
val_path   = f'./client_data/client_{client_id}_val.txt'

input_shape = [448, 448]
phi = 's'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class YOLOClient(fl.client.NumPyClient):

    def __init__(self, cid):
        self.round = 0
        self.cid = cid

        self.class_names, self.num_classes = get_classes(classes_path)

        # ======================
        # YOLO model
        # ======================
        self.model = YoloBody(input_shape, self.num_classes, phi, pretrained=False).to(device)
        self.yolo_loss = Loss(self.model)

        # ======================
        # dataset
        # ======================
        train_lines = open(train_path).readlines()
        val_lines = open(val_path).readlines()

        self.train_dataset = YoloDataset(
            train_lines, input_shape, self.num_classes,
            epoch_length=2,
            mosaic=True,
            mixup=True,
            mosaic_prob=0.5,
            mixup_prob=0.5,
            train=True,
            special_aug_ratio=0.7
        )

        self.val_dataset = YoloDataset(
            val_lines,
            input_shape,
            self.num_classes,
            epoch_length=1,
            mosaic=False,
            mixup=False,
            mosaic_prob=0,
            mixup_prob=0,
            train=False,
            special_aug_ratio=0
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=4,
            collate_fn=yolo_dataset_collate
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            collate_fn=yolo_dataset_collate
        )

        # ======================
        # optimizer
        # ======================
        self.Init_lr = 1e-2
        self.Min_lr = self.Init_lr * 0.01

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.Init_lr,
            momentum=0.937,
            weight_decay=5e-4
        )

        self.lr_scheduler_func = get_lr_scheduler(
            "cos",
            self.Init_lr,
            self.Min_lr,
            100
        )

        # ======================
        # logging
        # ======================
        self.log_dir = f'logs/client_{cid}_{datetime.datetime.now():%Y%m%d}'
        os.makedirs(self.log_dir, exist_ok=True)

        self.loss_history = LossHistory(self.log_dir, self.model, input_shape)

        self.eval_callback = EvalCallback(
            net=self.model,
            input_shape=input_shape,
            class_names=self.class_names,
            num_classes=self.num_classes,
            val_lines=val_lines,
            log_dir=self.log_dir,
            cuda=device.type == 'cuda',
            eval_flag=True,
            period=4,
            map_out_path=f".temp_map_out_client_{cid}"
        )

        # ======================
        # GAN components
        # ======================
        self.feat_dim = 512

        self.generator = DistributionGenerator(
            feat_dim=self.feat_dim,
            num_classes=self.num_classes
        ).to(device)

        self.discriminator = FeatureDiscriminator(
            feat_dim=self.feat_dim,
            num_classes=self.num_classes
        ).to(device)

        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)

        self.local_epochs = 6

    # ======================
    # FL: get parameters
    # ======================
    def get_parameters(self, config):

        params = []

        for v in self.model.state_dict().values():
            params.append(v.cpu().numpy())

        for v in self.generator.state_dict().values():
            params.append(v.cpu().numpy())


        return params

    # ======================
    # FL: set parameters
    # ======================
    def set_parameters(self, parameters):

        model_keys = list(self.model.state_dict().keys())
        gen_keys = list(self.generator.state_dict().keys())

        model_len = len(model_keys)
        gen_len = len(gen_keys)

        model_state = OrderedDict({
            k: torch.tensor(v).to(device)
            for k, v in zip(model_keys, parameters[:model_len])
        })

        gen_state = OrderedDict({
            k: torch.tensor(v).to(device)
            for k, v in zip(gen_keys, parameters[model_len:model_len + gen_len])
        })

        self.model.load_state_dict(model_state, strict=True)
        self.generator.load_state_dict(gen_state, strict=True)


    # ======================
    # training
    # ======================
    def fit(self, parameters, config):

        self.set_parameters(parameters)

        for epoch in range(self.local_epochs):

            set_optimizer_lr(self.optimizer, self.lr_scheduler_func, epoch)

            fit_one_epoch(
                model_train=self.model,
                model=self.model,
                ema=None,
                yolo_loss=self.yolo_loss,
                loss_history=self.loss_history,
                eval_callback=self.eval_callback,
                optimizer=self.optimizer,
                epoch=epoch,
                epoch_step=len(self.train_loader),
                epoch_step_val=len(self.val_loader),
                gen=self.train_loader,
                gen_val=self.val_loader,
                Epoch=self.local_epochs,
                cuda=device.type == 'cuda',
                fp16=False,
                scaler=None,
                save_period=1000,
                save_dir=self.log_dir,

                # ===== GAN =====
                generator=self.generator,
                discriminator=self.discriminator,
                optimizer_g=self.optimizer_g,
                optimizer_d=self.optimizer_d,

                feat_dim=self.feat_dim,
                base_align=0.1,
                base_contrast=0,
                base_gan=0.05,
                warmup_epoch=20,

                device=device,
                round=self.round
            )

            self.round += 1

        return self.get_parameters({}), len(self.train_dataset), {}

    # ======================
    # evaluation
    # ======================
    def evaluate(self, parameters, config):

        self.set_parameters(parameters)
        self.model.eval()

        loss_sum = 0
        n = 0

        with torch.no_grad():
            for imgs, targets in self.val_loader:

                imgs = imgs.to(device)
                targets = [t.to(device) for t in targets]

                loss = self.yolo_loss(self.model(imgs), targets)

                loss_sum += loss.item() * imgs.size(0)
                n += imgs.size(0)

        return loss_sum / max(n, 1), n, {}


if __name__ == "__main__":

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=YOLOClient(client_id)
    )