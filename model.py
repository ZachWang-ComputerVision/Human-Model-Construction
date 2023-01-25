import torch
import torch.nn as nn
import torchvision

class Block(nn.Module):
  def __init__(self, in_channels: int, out_channels: int, downsample: bool,
               upsample: bool, stride: int = 1, padding: int = 1):
    super(Block, self).__init__()

    intermediate_channels = in_channels // 2
    norm_layer = nn.BatchNorm2d
    self.bn0 = norm_layer(in_channels)

    self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=stride, padding=0)
    self.bn1 = norm_layer(intermediate_channels)
    
    self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=padding)
    self.bn2 = norm_layer(intermediate_channels)

    if not downsample:
      self.conv3 = nn.Conv2d(intermediate_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
      self.bn3 = norm_layer(out_channels)
    else:
      self.conv3 = nn.Conv2d(intermediate_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
      self.bn3 = norm_layer(out_channels)

    self.relu = nn.LeakyReLU(inplace=True)
    self.downsample = downsample
    self.pooling = nn.MaxPool2d(3, stride=2, padding=1)

    self.upsample = upsample
    # self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
    self.upsampling = nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1)

    # self.apply(self.weight_init)
    self.init_weights()

    # def weight_init(m):
      

    # self.pred = pred
    # self.sigmoid = nn.Sigmoid()
  
  def forward(self, x):

    if self.upsample:
      # x = self.upsampling(x)
      x = self.upsampling(x, output_size=torch.Size([x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2]))
      x = self.bn0(x)

    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv3(out)

    # if not self.pred:
    out = self.bn3(out)

    if self.downsample:
      identity = self.pooling(identity)

    out += identity
    out = self.relu(out)
    return out

  def init_weights(self):
    def weight_init(m):
      if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight)
        if getattr(m, 'bias') is not None:
          nn.init.constant_(m.bias, 0)
    self.apply(weight_init)
  



class Vertex2Image(nn.Module):
  def __init__(self, vertex_feature, input_channels=64):
    super(Vertex2Image, self).__init__()

    assert input_channels % 2 == 0
    assert vertex_feature.size(0) == input_channels
    half_input_channels = input_channels // 2

    # self.vertex_feature = vertex_feature
    # self.vertex_encoding = nn.Linear(n_vertex, n_vertex)
    # self.layer_norm = nn.LayerNorm(n_vertex)

    self.bridge_layer_1 = nn.Conv2d(6, half_input_channels, kernel_size=5, stride=1, padding=2)
    self.bn1 = nn.BatchNorm2d(half_input_channels)
    self.bridge_layer_2 = nn.Conv2d(half_input_channels, input_channels, kernel_size=5, stride=1, padding=2)
    self.bn2 = nn.BatchNorm2d(input_channels)
    self.concat_decode_256 = Block(input_channels, input_channels, downsample=False, upsample=False)
    self.bn3 = nn.BatchNorm2d(input_channels)
    
    self.relu = nn.LeakyReLU(inplace=True)

    # Vertex Decoder Component
    self.decode_256_128_1 = Block(input_channels, input_channels, downsample=True, upsample=False)
    self.decode_256_128_2 = Block(input_channels, input_channels, downsample=True, upsample=False)
    self.decode_256_128_3 = Block(input_channels, input_channels, downsample=True, upsample=False)
    self.decode_256_128_4 = Block(input_channels, input_channels, downsample=True, upsample=False)

    self.decode_128_64_1 = Block(input_channels, input_channels, downsample=True, upsample=False)
    self.decode_128_64_2 = Block(input_channels, input_channels, downsample=True, upsample=False)
    self.decode_128_64_3 = Block(input_channels, input_channels, downsample=True, upsample=False)

    self.decode_64_32_1 = Block(input_channels, input_channels, downsample=True, upsample=False)
    self.decode_64_32_2 = Block(input_channels, input_channels, downsample=True, upsample=False)
    
    self.decode_32_32 = Block(input_channels, input_channels, downsample=False, upsample=False)

    # Skip Connection Component
    self.connect_256_256_1 = Block(input_channels, input_channels, downsample=False, upsample=False)
    self.connect_256_256_2 = Block(input_channels, input_channels, downsample=False, upsample=False)
    self.connect_256_256_3 = Block(input_channels, input_channels, downsample=False, upsample=False)
    self.connect_256_256_4 = Block(input_channels, input_channels, downsample=False, upsample=False)
    self.connect_128_128_1 = Block(input_channels, input_channels, downsample=False, upsample=False)
    self.connect_128_128_2 = Block(input_channels, input_channels, downsample=False, upsample=False)
    self.connect_128_128_3 = Block(input_channels, input_channels, downsample=False, upsample=False)
    self.connect_64_64_1 = Block(input_channels, input_channels, downsample=False, upsample=False)
    self.connect_64_64_2 = Block(input_channels, input_channels, downsample=False, upsample=False)

    # Transition Layer
    self.transit_256_1 = nn.Conv2d(input_channels * 2, input_channels, kernel_size=3, stride=1, padding=1)
    self.transit_256_2 = nn.Conv2d(input_channels * 2, input_channels, kernel_size=3, stride=1, padding=1)
    self.transit_256_3 = nn.Conv2d(input_channels * 2, input_channels, kernel_size=3, stride=1, padding=1)
    self.transit_128_1 = nn.Conv2d(input_channels * 3, input_channels, kernel_size=3, stride=1, padding=1)
    self.transit_128_2 = nn.Conv2d(input_channels * 3, input_channels, kernel_size=3, stride=1, padding=1)
    self.transit_128_3 = nn.Conv2d(input_channels * 3, input_channels, kernel_size=3, stride=1, padding=1)
    self.transit_64_1 = nn.Conv2d(input_channels * 3, input_channels, kernel_size=3, stride=1, padding=1)
    self.transit_64_2 = nn.Conv2d(input_channels * 3, input_channels, kernel_size=3, stride=1, padding=1)
    self.transit_32 = nn.Conv2d(input_channels * 2, input_channels, kernel_size=3, stride=1, padding=1)
    self.transit_bn1 = nn.BatchNorm2d(input_channels)
    self.transit_bn2 = nn.BatchNorm2d(input_channels)
    self.transit_bn3 = nn.BatchNorm2d(input_channels)
    self.transit_bn4 = nn.BatchNorm2d(input_channels)
    self.transit_bn5 = nn.BatchNorm2d(input_channels)
    self.transit_bn6 = nn.BatchNorm2d(input_channels)
    self.transit_bn7 = nn.BatchNorm2d(input_channels)
    self.transit_bn8 = nn.BatchNorm2d(input_channels)
    self.transit_bn9 = nn.BatchNorm2d(input_channels)

    # self.transit_256_bn = 
    
    # Encoder Component
    self.encode_32_64_1 = Block(input_channels, input_channels, downsample=False, upsample=True)
    self.encode_32_64_2 = Block(input_channels, input_channels, downsample=False, upsample=True)
    self.encode_64_128_1 = Block(input_channels, input_channels, downsample=False, upsample=True)
    self.encode_64_128_2 = Block(input_channels, input_channels, downsample=False, upsample=True)
    self.encode_64_128_3 = Block(input_channels, input_channels, downsample=False, upsample=True)
    self.encode_128_256_1 = Block(input_channels, input_channels, downsample=False, upsample=True)
    self.encode_128_256_2 = Block(input_channels, input_channels, downsample=False, upsample=True)
    self.encode_128_256_3 = Block(input_channels, input_channels, downsample=False, upsample=True)
    self.encode_128_256_4 = Block(input_channels, input_channels, downsample=False, upsample=True)
    
    self.cat_encode_256 = Block(input_channels * 2, input_channels * 2, downsample=False, upsample=False)
    
    self.shrink = nn.Conv2d(input_channels * 2, half_input_channels, kernel_size=3, stride=1, padding=1)
    self.bn4 = nn.BatchNorm2d(half_input_channels)
    self.shrink_2 = nn.Conv2d(half_input_channels, half_input_channels, kernel_size=3, stride=1, padding=1)
    self.bn5 = nn.BatchNorm2d(half_input_channels)
    self.pred_fg = nn.Conv2d(half_input_channels, 3, kernel_size=3, stride=1, padding=1)
    self.sigmoid = nn.Sigmoid()

    self.init_weights()


  def init_weights(self):
      def weight_init(m):
        if isinstance(m, nn.Conv2d):
          nn.init.kaiming_normal_(m.weight)
          if getattr(m, 'bias') is not None:
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.ModuleList):
          for layer in m:
            # linear = layer[0]
            nn.init.kaiming_normal_(layer[0].weight)
      self.apply(weight_init)


  def forward(self, x):
    
    # n_batch_positions = torch.stack([
    #     torch.unsqueeze(self.positions * position_mask[i], dim=2) for i in range(x.size(0))
    #   ])
    # n_batch_positions = self.positional_encoding(n_batch_positions)
    # print("n_batch_positions: ", n_batch_positions.shape)
    # n_batch_positions = torch.permute(n_batch_positions, (0, 3, 1, 2))
    # assert x.size() == v.size()

    # vertex_encoding = self.vertex_encoding(self.vertex_feature)   # input_channel x 6890
    # vertex_encoding = self.layer_norm(vertex_encoding)
    # vertex_encoding = vertex_encoding.T

    # B, H, W = mask.size()

    # vertex_feature_map = torch.zeros((B, vertex_encoding.size(1), H, W), dtype=torch.float32)

    # for b, batch_item in enumerate(mask):
    #   for i, row_item in enumerate(batch_item):
    #     for j, item in enumerate(row_item):
    #       if torch.isnan(item) == False:
    #         vertex_feature_map[b, :, i, j] = vertex_encoding[int(item)]

    # if cuda:
    #   vertex_feature_map = vertex_feature_map.cuda()

    x = self.bridge_layer_1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.bridge_layer_2(x)
    x = self.bn2(x)
    x = self.relu(x)
    x = self.concat_decode_256(x)
    x = self.bn3(x)


    deconv_256_128_1 = self.decode_256_128_1(x)
    deconv_128_64_1 = self.decode_128_64_1(deconv_256_128_1)
    deconv_64_32_1 = self.decode_64_32_1(deconv_128_64_1)

    connect_256_256_1 = self.connect_256_256_1(x)
    connect_128_128_1 = self.connect_128_128_1(deconv_256_128_1)
    connect_64_64_1 = self.connect_64_64_1(deconv_128_64_1)
    deconv_32_32 = self.decode_32_32(deconv_64_32_1)

    encode_32_64_1 = self.encode_32_64_1(deconv_64_32_1)
    encode_64_128_1 = self.encode_64_128_1(deconv_128_64_1)
    encode_128_256_1 = self.encode_128_256_1(deconv_256_128_1)

    cat_256_1 = torch.cat( (encode_128_256_1, connect_256_256_1), 1 )

    cat_256_1 = self.transit_bn1(self.transit_256_1(cat_256_1))

    deconv_256_128_2 = self.decode_256_128_2(cat_256_1)
    cat_128_1 = torch.cat( (encode_64_128_1, connect_128_128_1, deconv_256_128_2), 1 )
    cat_128_1 = self.transit_bn4(self.transit_128_1(cat_128_1))

    deconv_128_64_2 = self.decode_128_64_2(cat_128_1)
    cat_64_1 = torch.cat( (encode_32_64_1, connect_64_64_1, deconv_128_64_2), 1 )
    cat_64_1 = self.transit_bn7(self.transit_64_1(cat_64_1))

    deconv_64_32_2 = self.decode_64_32_2(cat_64_1)
    cat_32 = torch.cat( (deconv_32_32, deconv_64_32_2), 1 )
    cat_32 = self.transit_bn9(self.transit_32(cat_32))
    


    connect_256_256_2 = self.connect_256_256_2(cat_256_1)
    connect_128_128_2 = self.connect_128_128_2(cat_128_1)
    connect_64_64_2 = self.connect_64_64_2(cat_64_1)
    
    encode_128_256_2 = self.encode_128_256_2(cat_128_1)
    encode_64_128_2 = self.encode_64_128_2(cat_64_1)
    encode_32_64_2 = self.encode_32_64_2(cat_32)
    
    cat_256_2 = torch.cat( (encode_128_256_2, connect_256_256_2), 1 )

    cat_256_2 = self.transit_bn2(self.transit_256_2(cat_256_2))

    deconv_256_128_3 = self.decode_256_128_3(cat_256_2)
    cat_128_2 = torch.cat( (encode_64_128_2, connect_128_128_2, deconv_256_128_3), 1 )
    cat_128_2 = self.transit_bn5(self.transit_128_2(cat_128_2))

    deconv_128_64_3 = self.decode_128_64_3(cat_128_2)
    cat_64_2 = torch.cat( (encode_32_64_2, connect_64_64_2, deconv_128_64_3), 1 )
    cat_64_2 = self.transit_bn8(self.transit_64_2(cat_64_2))
    


    connect_256_256_3 = self.connect_256_256_3(cat_256_2)
    connect_128_128_3 = self.connect_128_128_3(cat_128_2)

    encode_64_128_3 = self.encode_64_128_3(cat_64_2)
    encode_128_256_3 = self.encode_128_256_3(cat_128_2)

    cat_256_3 = torch.cat( (encode_128_256_3, connect_256_256_3), 1 )
    cat_256_3 = self.transit_bn3(self.transit_256_3(cat_256_3))

    deconv_256_128_4 = self.decode_256_128_4(cat_256_3)
    cat_128_3 = torch.cat( (encode_64_128_3, connect_128_128_3, deconv_256_128_4), 1 )
    cat_128_3 = self.transit_bn6(self.transit_128_3(cat_128_3))

    connect_256_256_4 = self.connect_256_256_4(cat_256_3)
    encode_128_256_4 = self.encode_128_256_4(cat_128_3)

    cat_256_4 = torch.cat( (encode_128_256_4, connect_256_256_4), 1 )
    cat_encode_256 = self.cat_encode_256(cat_256_4)

    out = self.shrink(cat_encode_256)
    out = self.bn4(out)
    out = self.shrink_2(out)
    out = self.bn5(out)
    out = self.pred_fg(out)
    out = self.sigmoid(out)

    out = out * 255

    return out

