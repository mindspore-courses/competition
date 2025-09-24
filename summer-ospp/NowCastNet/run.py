import argparse
import mindspore as ms
from nowcasting.models.model_factory import Model
import nowcasting.evaluator as evaluator


parser = argparse.ArgumentParser(description='NowcastNet MindSpore')

parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--worker', type=int, default=1)
parser.add_argument('--cpu_worker', type=int, default=1)
parser.add_argument('--dataset_name', type=str, default='radar')
parser.add_argument('--input_length', type=int, default=9)
parser.add_argument('--total_length', type=int, default=29)
parser.add_argument('--img_height', type=int, default=512)
parser.add_argument('--img_width', type=int, default=512)
parser.add_argument('--img_ch', type=int, default=2)
parser.add_argument('--case_type', type=str, default='normal')
parser.add_argument('--model_name', type=str, default='NowcastNet')
parser.add_argument('--gen_frm_dir', type=str, default='results/nowcasting_ms')
parser.add_argument('--evo_pretrained_model', type=str, default="/home/summer/mindspore/code_mindspore/ckpt/evo/evolution.ckpt")
parser.add_argument('--gen_pretrained_model', type=str, default="/home/summer/mindspore/code_mindspore/ckpt/gen/generator.ckpt")
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_save_samples', type=int, default=10)
parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--noise_scale', type=int, default=32)
parser.add_argument('--dataset_path', type=str, default='/home/summer/mindspore/code_mindspore/dataset')
parser.add_argument('--data_frequency', type=int, default=10)


args = parser.parse_args()

args.evo_ic = args.total_length - args.input_length
args.gen_oc = args.total_length - args.input_length
args.ic_feature = args.ngf * 10

ms.set_context(device_target="CPU")

model = Model(args)
evaluator.infer(model, args)