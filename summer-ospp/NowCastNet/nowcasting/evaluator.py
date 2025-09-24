import mindspore as ms
from nowcasting.data_provider import datasets_factory
from nowcasting.layers.utils import plt_img

def infer(model, configs):
    test_dataset_generator = datasets_factory.RadarData(configs, module_name="generation")
    test_dataset = datasets_factory.NowcastDataset(dataset_generator=test_dataset_generator, module_name="generation")
    test_dataset = test_dataset.create_dataset(configs.batch_size)
    data = next(test_dataset.create_dict_iterator())
    inp, evo_result, labels = data.get("inputs"), data.get("evo"), data.get("labels")
    noise_scale = configs.noise_scale
    batch_size = configs.batch_size
    w_size = configs.img_width
    h_size = configs.img_height
    ngf = configs.ngf
    noise = ms.tensor(ms.numpy.randn((batch_size, ngf, h_size // noise_scale, w_size // noise_scale)), inp.dtype)
    pred = model.network.gen_net(inp, evo_result, noise)
    plt_img(field=pred[0].asnumpy(), label=labels[0].asnumpy(), fig_name="./generation_example.png")