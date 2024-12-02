# run_model.py
import torch
from src.networks.vit_seg_modeling import VisionTransformer, CONFIGS

def print_shape(tensor, layer_name):
    print(f"{layer_name} output shape: {tensor.shape}")

# 选择配置
config_name = 'R50-ViT-B_16'
config = CONFIGS[config_name]

# 创建模型实例
model = VisionTransformer(config, img_size=224, num_classes=21843, zero_head=False, vis=False)

# 创建一个随机输入张量，假设输入图像大小为224x224，批量大小为1
input_tensor = torch.randn(1, 3, 224, 224)

# 通过模型并打印形状
print("Input shape:", input_tensor.shape)
embedding_output, features = model.transformer.embeddings(input_tensor)
print_shape(embedding_output, "Embeddings")

# 通过Encoder部分
encoded, attn_weights = model.transformer.encoder(embedding_output)
print_shape(encoded, "Encoder")

# 通过Decoder部分
decoder_output = model.decoder(encoded, features)
print_shape(decoder_output, "Decoder")

# 通过SegmentationHead部分
logits = model.segmentation_head(decoder_output)
print_shape(logits, "SegmentationHead")