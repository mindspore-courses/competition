# Mixtral模型优化方法：

## 1、	使用mint替换ops算子

```
        causal_mask *= ops.arange(target_length) > cache_position.reshape(-1, 1)

        concatenated_gate_logits = ops.cat(list(gate_logits), dim=0)
        ......
```

改为：

```
        causal_mask *= mint.arange(target_length) > cache_position.reshape(-1, 1)

        concatenated_gate_logits = mindspore.mint.cat(list(gate_logits), dim=0)
        ......
```

收益较小，平均1-2ms的提升

## 2、	使用RMSnorm融合算子替换小算子

源码：

```
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(mindspore.float32)
        variance = ops.mean(hidden_states.pow(2), -1, keepdim=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```

改为：

```
        return F.rms_norm(hidden_states, self.weight, self.variance_epsilon)
```
## 3、	将attention的q、k、v linear拼接后展开

源码：

```
        self.q_tproj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        ......
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
```

改为：

```
        self.w_qkv = nn.Linear(self.hidden_size, self.num_heads * self.head_dim + self.num_key_value_heads * self.head_dim * 2, bias=False)
        ......
        qkv = self.cast(self.w_qkv(hidden_states), mindspore.float16)
        # [batch_size, seq_len, num_heads, head_dim] 
        query_states, key_states, value_states = self.split_qkv(qkv, (self.hidden_size,  self.num_key_value_heads * self.head_dim,  self.num_key_value_heads * self.head_dim), 2)
```