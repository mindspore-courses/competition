# import torch
import mindspore as ms
# ª´¢±?«Á?ª£¡ëª´ª¤ª®¡ë«Øtorch¡ë¢Ä?ªÔmindsporeª©¢ìckpt??ª©¢Ä¢îª¤¢í
# def print_torch_ckpt(ckpt_path):
#     print(f"===== PyTorch ckpt ??¢Ä«Ô ({ckpt_path}) =====")
#     torch_ckpt = torch.load(ckpt_path, map_location="cpu")

#     if "model" in torch_ckpt:
#         state_dict = torch_ckpt["model"]
#     elif "state_dict" in torch_ckpt:
#         state_dict = torch_ckpt["state_dict"]
#     else:
#         state_dict = torch_ckpt

#     for name, param in state_dict.items():
#         shape = tuple(param.shape) if hasattr(param, "shape") else ()
#         print(f"{name}: {shape}")


def print_ms_ckpt(ckpt_path):
    print(f"===== MindSpore ckpt ??¢Ä«Ô ({ckpt_path}) =====")
    ms_ckpt = ms.load_checkpoint(ckpt_path)

    for name, param in ms_ckpt.items():
        shape = tuple(param.data.shape)
        print(f"{name}: {shape}")


if __name__ == "__main__":
    # ?¢ğ?©ĞªÁ¢í?ª¤
    # torch_ckpt_path = "model_000014.ckpt"
    ms_ckpt_path = "/home/outbreak/mindspore/MVSNet_pytorch/model_000014ms.ckpt"
    print(ms.__version__)
    print(ms.get_context("device_target"))  # «¢ªÔ? "GPU"
    # print_torch_ckpt(torch_ckpt_path)
    # print()
    print_ms_ckpt(ms_ckpt_path)
