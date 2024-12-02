# run_model.py
import argparse

import numpy as np
import torch
from src.trans_unet.vit_seg_modeling import VisionTransformer, CONFIGS

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

config_vit = CONFIGS[args.vit_name]
config_vit.n_classes = args.num_classes
config_vit.n_skip = args.n_skip
config_vit.pretrained_path = '../../project_TransUNet/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
if args.vit_name.find('R50') != -1:
    config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
net = VisionTransformer(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
net.load_from(weights=np.load(config_vit.pretrained_path))


def print_shape(tensor, layer_name):
    print(f"{layer_name} output shape: {tensor.shape}")


# 创建一个随机输入张量，假设输入图像大小为224x224，批量大小为1
input_tensor = torch.randn(1, 3, 480, 480).cuda()  # 将输入数据移动到GPU

# # 通过模型并打印形状
# print("Input shape:", input_tensor.shape)
# embedding_output, features = net.transformer.embeddings(input_tensor)
# print_shape(embedding_output, "Embeddings")
#
# # 通过Encoder部分
# encoded, attn_weights = net.transformer.encoder(embedding_output)
# print_shape(encoded, "Encoder")
#
# print("\n\n\n\n\n==========================")
#
# # 通过Decoder部分
# decoder_output = net.decoder(encoded, features)
# print_shape(decoder_output, "Decoder")
#
# # 通过SegmentationHead部分
# logits = net.segmentation_head(decoder_output)
# print_shape(logits, "SegmentationHead")


# 通过模型并打印形状
print("Input shape:", input_tensor.shape)

# 通过整个网络
logits = net(input_tensor)

# 打印最终输出的形状
print_shape(logits, "Final Output")