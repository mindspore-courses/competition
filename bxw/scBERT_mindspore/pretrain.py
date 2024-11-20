import argparse
from dataset1 import load_data
from performer_mindspore import PerformerLM
from mindspore.nn import Adam, CrossEntropyLoss
from tqdm import tqdm
from mindspore import ops, save_checkpoint, Tensor
import math
from functools import reduce
import mindspore as ms
from mindspore import value_and_grad, ParallelMode, nn
from mindspore.communication import init
from mindspore import Profiler
import os

model = None
loss_fn = None
# get the random prob matrix and True means smaller than prob threshold
def prob_mask_like(t, prob):
    return ops.uniform(t.shape, Tensor(0, dtype=ms.float32), Tensor(1, dtype=ms.float32)).float() < prob

# get the mask matrix which cannot be masked
def mask_with_tokens(t, token_ids):
    # print(ops.Unique()(t[0]))
    init_no_mask = ops.full_like(t, False, dtype=ms.uint8)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return Tensor(mask, dtype=ms.uint8)

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len = mask.shape
    max_masked = math.ceil(prob * seq_len)      # num of mask of a single sequence in average
    num_tokens = mask.sum(axis=-1, keepdims=True)     # num of pure tokens of each sequence except special tokens
    mask_excess = ops.cat((ops.zeros(size=(batch), dtype=ms.float32), ops.arange(1, seq_len,dtype=ms.float32).repeat(batch))).reshape(batch,seq_len)
    #print(mask_excess.shape)
    mask_excess = (mask_excess >= (num_tokens * prob).ceil())        # only 15% of pure tokens can be masked
    mask_excess = ops.Reshape()(mask_excess, (batch, seq_len))
    mask_excess = mask_excess[:, :max_masked]       # get difference between 15% of pure tokens and 15% of all tokens
    rand = ops.rand((batch, seq_len)).masked_fill(~mask, -1e9)     # rand (0-1) as prob, special token use -1e9
    _, sampled_indices = rand.topk(max_masked, dim=-1)      # get index of topk prob to mask
    #print(sampled_indices.shape)
    sampled_indices = (sampled_indices + 1).masked_fill(mask_excess, 0)        # delete difference of mask not pure
    #print(sampled_indices.shape)
    new_mask = ops.zeros((batch, seq_len + 1), dtype=ms.uint8)     # get (batch, seq_len) shape zero matrix
    new_mask = new_mask.scatter(-1, sampled_indices, ops.ones(shape=ops.shape(sampled_indices), dtype=ms.uint8))    # set masks in zero matrix as 1
    new_mask = ops.Cast()(new_mask, ms.uint8)
    return new_mask[:, 1:]      # the final mask, True is mask

def data_mask(data,
    mask_prob=None,
    replace_prob=None,
    num_tokens=None,
    random_token_prob=None,
    mask_token_id=None,
    pad_token_id=None,
    mask_ignore_token_ids=None
):
    global MASK_PROB, REPLACE_PROB, RANDOM_TOKEN_PROB, MASK_TOKEN_ID, PAD_TOKEN_ID, MASK_IGNORE_TOKEN_IDS
    replace_prob = REPLACE_PROB
    mask_prob= MASK_PROB
    random_token_prob = RANDOM_TOKEN_PROB
    mask_token_id = MASK_TOKEN_ID
    pad_token_id = PAD_TOKEN_ID
    mask_ignore_token_ids = MASK_IGNORE_TOKEN_IDS
    
    mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])
    # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
    # also do not include these special tokens in the tokens chosen at random
    no_mask = mask_with_tokens(data, mask_ignore_token_ids)   # ignore_token as True, will not be masked later
    mask = get_mask_subset_with_prob(~no_mask, mask_prob)      # get the True/False mask matrix
    # get mask indices
    ## mask_indices = torch.nonzero(mask, as_tuple=True)   # get the index of mask(nonzero value of mask matrix)
    # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)
    masked_input = data
    # if random token probability > 0 for mlm
    if random_token_prob > 0:
        assert num_tokens is not None, 'num_tokens keyword must be supplied when instantiating MLM if using random token replacement'
        random_token_prob = prob_mask_like(data, random_token_prob)       # get the mask matrix of random token replace
        random_tokens = ops.randint(0, num_tokens, data.shape)     # generate random token matrix with the same shape as input
        random_no_mask = mask_with_tokens(random_tokens, mask_ignore_token_ids)        # not masked matrix for the random token matrix
        random_token_prob &= ~random_no_mask        # get the pure mask matrix of random token replace
        random_indices = ops.nonzero(random_token_prob, as_tuple=True)        # index of random token replace
        masked_input[random_indices] = random_tokens[random_indices]        # replace some tokens by random token
    # [mask] input
    replace_prob = prob_mask_like(data, replace_prob)     # get the mask matrix of token being masked
    masked_input = masked_input.masked_fill(ops.Cast()(mask * replace_prob, ms.bool_), mask_token_id)        # get the data has been masked by mask_token
    # mask out any tokens to padding tokens that were not originally going to be masked
    labels = data.masked_fill(~mask, pad_token_id)        # the label of masked tokens
    return masked_input, labels

def cum_loss_and_logits(data, label):
    global model, loss_fn, SEQ_LEN
    logits = model(data)
    label = ops.repeat_elements(label, rep=7, axis=-1)
    label = ops.reshape(label, (-1, SEQ_LEN, 7))
    # label = ops.Cast()(label, dtype=ms.dtype.float32)
    loss = loss_fn(logits, label)
    return loss, logits

def build_model(args):
    global CLASS, SEQ_LEN, POS_EMBED_USING, model
    model = PerformerLM(
            num_tokens = CLASS,         # 7
            dim = 200,
            depth = 6,
            max_seq_len = SEQ_LEN,      # 16907
            heads = 10,
            # local_attn_heads = 0,
    )
    print("build model success.")
    count = sum([ item.size for item in model.get_parameters()])
    names = [item.name for item in model.trainable_params()]
    print("param count is {}, names: {}, count: {}".format(count, str(names), len(names)))
    
    if args.enable_pipeline:
        model.init_pipeline(0)
        model.performer.layers[0].init_pipeline(1)
        model.performer.layers[0].attention.init_pipeline(1)
    return

def build_optimizer_and_scheduler(model):
    global LEARNING_RATE, PAD_TOKEN_ID, loss_fn, optimizer
    # optimizer
    optimizer = Adam(params=model.trainable_params(), learning_rate=LEARNING_RATE)
    loss_fn = CrossEntropyLoss(ignore_index = PAD_TOKEN_ID, reduction='mean')
    print("build optimizer success.")
    return optimizer

def train_one_epoch(train_dataloader, grad_fn, optimizer):
    print("===========================")
    global PAD_TOKEN_ID, model
    running_loss = 0.0
    cum_acc = 0.0
    model.set_train(True)
    for _, (data,) in enumerate(tqdm(train_dataloader.create_tuple_iterator())):
        # forward 推理
        # profiler = Profiler(output_path="./output")
        data, labels = data_mask(data)
        # labels = ops.zeros(size=(1, 16907), dtype=ms.float32)
        labels = ops.cast(labels, ms.float32)
        (loss, logits), grads = grad_fn(data, labels)
        optimizer(grads)
        # 累加损失
        running_loss += loss.item() 
        # 计算精度
        final = ops.softmax(logits, axis=-1)[..., 1:-1]
        final = final.argmax(axis=-1) + 1
        pred_num = (labels != PAD_TOKEN_ID).sum(axis=-1)
        correct_num = ops.mul(Tensor(labels != PAD_TOKEN_ID, dtype=ms.uint8), Tensor(final == labels, dtype=ms.uint8)).sum(axis=-1)
        cum_acc += ops.true_divide(correct_num, pred_num).mean().item()
        del data, labels, final
        # profiler.analyse()
        
    return running_loss, cum_acc

def eval_one_epoch(val_dataloader):
    global PAD_TOKEN_ID, loss_fn, model, SEQ_LEN
    model.set_train(False)
    predictions = []
    truths = []
    running_loss = 0.0
    print("========== 开始验证")
    correct_num = 0
    val_num = 0
    for _, (data,) in enumerate(tqdm(val_dataloader.create_tuple_iterator())):
        data, ori_labels = data_mask(data)
        ori_labels = ops.cast(ori_labels, ms.float32)
        logits = model(data)
        labels = ops.repeat_elements(ori_labels, rep=7, axis=-1)
        labels = ops.reshape(labels, (-1, SEQ_LEN, 7))
        labels = ops.cast(labels, dtype=ms.float32)
        loss = loss_fn(logits, labels)
        running_loss += loss.item()
        final = ops.softmax(logits, axis=-1)[..., 1:-1]
        final = final.argmax(axis=-1) + 1
        correct_num += ops.mul(Tensor(ori_labels!=PAD_TOKEN_ID, dtype=ms.uint8), Tensor(final == ori_labels, dtype=ms.uint8)).sum(axis=-1).sum()
        val_num += Tensor(ori_labels != PAD_TOKEN_ID, dtype=ms.uint8).sum(axis=-1).sum()
        del data, labels, logits, final, ori_labels
    val_loss = running_loss / len(val_dataloader)
    val_acc = 100 * correct_num / val_num
    del predictions, truths
    return val_loss, val_acc

def train(optimizer, train_dataloader, val_dataloader):
    global EPOCHS,VALIDATE_EVERY, MODEL_NAME, loss_fn
    
    train_num_step =  len(train_dataloader)
    grad_fn = value_and_grad(cum_loss_and_logits, grad_position=None, weights=model.trainable_params(), has_aux=True)
    for epoch in range(EPOCHS):
        running_loss, cum_acc = train_one_epoch(train_dataloader, grad_fn, optimizer)
        # log epoch的信息
        epoch_loss = running_loss / train_num_step
        epoch_acc = 100 * cum_acc / train_num_step
        log_string = f'    ==  Epoch: {epoch} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  =='
        print(log_string)
        with open('pretrain_result.txt', 'a') as f:  
            f.write(log_string  + '\n')

        # 进行一次验证
        if epoch % VALIDATE_EVERY == 0:
            val_loss, val_acc = eval_one_epoch(val_dataloader)
            log_string = f'    ==  Epoch: {epoch} | Validation Loss: {val_loss} | Accuracy: {val_acc.item()}%  =='
            print(log_string)
            with open('pretrain_result.txt', 'a') as f:  
                f.write(log_string  + '\n')

        # 存模型
        ckpt_dir = "./" + PRETRAIN_PATH
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_file = f"pretrain-{epoch}.ckpt"
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)
        save_checkpoint(model, ckpt_path)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable_pipeline", type=bool, default=False, help='Local process rank.')
    parser.add_argument("--device_id", type=int, default=-1, help='Local process rank.')
    parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
    parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
    parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
    parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
    parser.add_argument("--batch_size", type=int, default=3, help='Number of batch size.')
    parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
    parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
    parser.add_argument("--mask_prob", type=float, default=0.15, help='Probability of masking.')
    parser.add_argument("--replace_prob", type=float, default=0.9, help='Probability of replacing with [MASK] token for masking.')
    parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
    parser.add_argument("--data_path", type=str, default='./data/panglao_10000.h5ad', help='Path of data for pretraining.')
    parser.add_argument("--ckpt_dir", type=str, default='./pretrain_ckpts/', help='Directory of checkpoint to save.')
    parser.add_argument("--model_name", type=str, default='panglao', help='Pretrained model name.')
    args = parser.parse_args()
    return args
            
if __name__ == "__main__":
    # 1. 解析命令行参数
    args = parse()
    if args.enable_pipeline:
        ms.set_context(mode=0, device_target="Ascend")
        ms.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, pipeline_stages=2, pipeline_result_broadcast=True)
        init()
        ms.set_seed(1)
    else:
        ms.set_context(variable_memory_max_size='29GB')
        ms.set_context(mode=0, device_target="Ascend", device_id=0)
    # 2. 声明全局变量，方便使用
    
    SEED = args.seed
    EPOCHS = args.epoch
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    SEQ_LEN = args.gene_num + 1
    # SEQ_LEN = 15000
    VALIDATE_EVERY = args.valid_every
    CLASS = args.bin_num + 2
    MASK_PROB = args.mask_prob
    REPLACE_PROB = args.replace_prob
    RANDOM_TOKEN_PROB = 0.
    MASK_TOKEN_ID = CLASS - 1
    PAD_TOKEN_ID = CLASS - 1
    MASK_IGNORE_TOKEN_IDS = [0]
    POS_EMBED_USING = args.pos_embed
    MODEL_NAME = args.model_name
    PRETRAIN_PATH = args.ckpt_dir
    
    # 3. 加载数据集
    train_dataloader, val_dataloader = load_data(args.data_path, CLASS, SEED, BATCH_SIZE, SEQ_LEN)
    
    # 4. 加载模型
    build_model(args)
    
    # 4. 构建优化器和调度器, 损失函数、归一化操作
    optimizer = build_optimizer_and_scheduler(model)
    
    # 5. 开始训练
    train(optimizer, train_dataloader, val_dataloader)
