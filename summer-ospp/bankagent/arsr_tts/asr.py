# asr_system.py
import os
import numpy as np
import librosa
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore.dataset import GeneratorDataset
from mindspore.train import Model, LossMonitor
from mindspore.common.initializer import XavierUniform

# ======================
# 1. 数据预处理模块
# ======================
class LJSpeechDataset:
    """LJSpeech数据集加载与特征提取"""
    def __init__(self, data_path, sr=16000, n_mfcc=40):
        self.wav_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.wav')])
        self.transcripts = self._load_transcripts()
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.char2idx = {'a': 0, 'b': 1, ...}  # 实际使用时需要完整字符集
        self.idx2char = {v: k for k, v in self.char2idx.items()}

    def _load_transcripts(self):
        """加载文本转录"""
        transcripts = []
        for wav_file in self.wav_files:
            txt_file = wav_file.replace('.wav', '.txt')
            with open(txt_file, 'r') as f:
                transcripts.append(f.read().strip())
        return transcripts

    def __getitem__(self, index):
        """提取MFCC特征和文本标签"""
        # 音频处理
        wav, _ = librosa.load(self.wav_files[index], sr=self.sr)
        mfcc = librosa.feature.mfcc(y=wav, sr=self.sr, n_mfcc=self.n_mfcc)
        
        # 文本处理
        text = self.transcripts[index]
        token_ids = [self.char2idx[c] for c in text if c in self.char2idx]
        
        return mfcc.T, np.array(token_ids, dtype=np.int32)

    def __len__(self):
        return len(self.wav_files)

# ======================
# 2. 模型架构 (Transformer-based)
# ======================
class TransformerASR(nn.Cell):
    """基于Transformer的ASR模型"""
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        
        # 音频特征编码
        self.src_embed = nn.Dense(40, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer主干
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 文本解码
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=1024)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # 输出层
        self.fc_out = nn.Dense(d_model, vocab_size, weight_init=XavierUniform())
        
    def construct(self, src, tgt):
        # src: [batch, src_len, 40]
        # tgt: [batch, tgt_len]
        
        # 编码器处理音频特征
        src = self.src_embed(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        
        # 解码器生成文本
        tgt = self.tgt_embed(tgt) * np.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, memory)
        
        return self.fc_out(output)

class PositionalEncoding(nn.Cell):
    """Transformer位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = Tensor(pe, ms.float32)
        
    def construct(self, x):
        return x + self.pe[:x.shape[1]]

# ======================
# 3. 训练流程
# ======================
def train_asr():
    # 环境设置
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    
    # 数据准备
    dataset = LJSpeechDataset("data/LJSpeech-1.1")
    train_data = GeneratorDataset(dataset, ["features", "labels"])
    train_data = train_data.batch(32)
    
    # 模型初始化
    model = TransformerASR(vocab_size=len(dataset.char2idx))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=0.0001)
    
    # 定义训练网络
    net = nn.WithLossCell(model, loss_fn)
    train_net = nn.TrainOneStepCell(net, optimizer)
    
    # 训练模型
    model = Model(train_net)
    model.train(epoch=10, 
               train_dataset=train_data,
               callbacks=[LossMonitor()])

    # 保存模型
    ms.save_checkpoint(model, "asr_model.ckpt")

# ======================
# 4. 推理部署
# ======================
class ASRPredictor:
    """ASR预测接口"""
    def __init__(self, ckpt_path, char2idx):
        self.model = TransformerASR(vocab_size=len(char2idx))
        ms.load_checkpoint(ckpt_path, self.model)
        self.model.set_train(False)
        self.char2idx = char2idx
        self.idx2char = {v: k for k, v in char2idx.items()}
        
    def predict(self, wav_path):
        # 特征提取
        mfcc = self._extract_features(wav_path)
        src = Tensor(mfcc[np.newaxis, ...], ms.float32)
        
        # 自回归解码
        tgt = Tensor([[self.char2idx['<s>']]], ms.int32)  # 开始符
        output_ids = []
        
        for _ in range(100):  # 最大生成长度
            outputs = self.model(src, tgt)
            next_id = int(outputs[0, -1].argmax())
            if next_id == self.char2idx['</s>']:  # 结束符
                break
            output_ids.append(next_id)
            tgt = ops.concat((tgt, Tensor([[next_id]], ms.int32)), axis=1)
            
        return ''.join([self.idx2char[i] for i in output_ids])
    
    def _extract_features(self, wav_path):
        wav, _ = librosa.load(wav_path, sr=16000)
        return librosa.feature.mfcc(y=wav, sr=16000, n_mfcc=40).T

# ======================
# 5. 主程序入口
# ======================
if __name__ == "__main__":
    # 训练模型
    train_asr()
    
    # 测试推理
    predictor = ASRPredictor("asr_model.ckpt", char2idx={'a':0, 'b':1, ...})
    result = predictor.predict("test.wav")
    print("识别结果:", result)
