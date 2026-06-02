# server.py
# CUDA_VISIBLE_DEVICES=0 nohup python server.py > server.log 2>&1 &

import os
# os.environ["GRPC_ARG_KEEPALIVE_TIME_MS"] = "60000"
# os.environ["GRPC_ARG_KEEPALIVE_TIMEOUT_MS"] = "5000"
# os.environ["GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS"] = "60000"
# os.environ["GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA"] = "0"

import flwr as fl
import torch
from flwr.common import parameters_to_ndarrays

# ====== ⚠️ 引入 YOLO 模型，仅用于 state_dict 结构 ======
from nets.yolo import YoloBody


# ----------------------------
# 原来的 metrics_avg（不动）
# ----------------------------
def metrics_avg(metrics):
    total_samples = sum(num for num, _ in metrics)
    loss_sum = sum(num * m.get("loss", 0.0) for num, m in metrics)
    return {"loss": loss_sum / total_samples}


# ============================
# 自定义 FedAvg（加保存逻辑）
# ============================
class FedAvgWithSave(fl.server.strategy.FedAvg):
    def __init__(self, model, save_every=10, save_dir="global_models", **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.save_every = save_every
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            rnd, results, failures
        )

        # -------- 每 save_every 轮保存一次 --------
        if aggregated_parameters is not None and rnd % self.save_every == 0:
            self._save_global_model(aggregated_parameters, rnd)

        return aggregated_parameters, aggregated_metrics

    def _save_global_model(self, parameters, rnd):
        params_ndarrays = parameters_to_ndarrays(parameters)

        state_dict = {
            k: torch.tensor(v)
            for k, v in zip(self.model.state_dict().keys(), params_ndarrays)
        }

        save_path = os.path.join(
            self.save_dir, f"global_round_{rnd}.pth"
        )
        torch.save(state_dict, save_path)

        print(f"[Server] ✅ Saved global model at round {rnd}", flush=True)


# ============================
# main
# ============================
def main():
    # -------- 必须与 client 保持一致 --------
    input_shape = [448, 448]
    num_classes = 4        # ⚠️ 改成你自己的
    phi = "s"

    # dummy model：只用于参数结构
    dummy_model = YoloBody(
        input_shape=input_shape,
        num_classes=num_classes,
        phi=phi,
        pretrained=False
    )

    strategy = FedAvgWithSave(
        model=dummy_model,
        save_every=10,                 # ✅ 每 10 轮保存
        save_dir="global_models_e6",

        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=2,
        min_available_clients=3,
        fit_metrics_aggregation_fn=metrics_avg,
        evaluate_metrics_aggregation_fn=metrics_avg,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(
            num_rounds=100,
            round_timeout=3000
        ),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
