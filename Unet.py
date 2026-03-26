from basic_blocks import *


class UnetGenerator(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(UnetGenerator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = 64
        act_fn = nn.ReLU()

        self.down_1 = conv_block_2(self.in_dim, self.num_filter, act_fn)
        self.pool_1 = maxpool()
        self.down_2 = conv_block_2(self.num_filter * 1, self.num_filter * 2, act_fn)
        self.pool_2 = maxpool()
        self.down_3 = conv_block_2(self.num_filter * 2, self.num_filter * 4, act_fn)
        self.pool_3 = maxpool()
        self.down_4 = conv_block_2(self.num_filter * 4, self.num_filter * 8, act_fn)
        self.pool_4 = maxpool()

        self.bridge = conv_block_2(self.num_filter * 8, self.num_filter * 16, act_fn)

        self.trans_1 = conv_trans_block(self.num_filter * 16, self.num_filter * 8)
        self.up_1 = conv_block_2(self.num_filter * 16, self.num_filter * 8, act_fn)
        self.trans_2 = conv_trans_block(self.num_filter * 8, self.num_filter * 4)
        self.up_2 = conv_block_2(self.num_filter * 8, self.num_filter * 4, act_fn)
        self.trans_3 = conv_trans_block(self.num_filter * 4, self.num_filter * 2)
        self.up_3 = conv_block_2(self.num_filter * 4, self.num_filter * 2, act_fn)
        self.trans_4 = conv_trans_block(self.num_filter * 2, self.num_filter * 1)
        self.up_4 = conv_block_2(self.num_filter * 2, self.num_filter * 1, act_fn)

        self.out = nn.Conv2d(self.num_filter, self.out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, inputImage, secondImage=None, threshold_down1=0.4, threshold_down2=0.6,
                threshold_down3=0.8, threshold_down4=1.0, threshold_bridge=1.2):
        down_1_first = self.down_1(inputImage)
        pool_1_first = self.pool_1(down_1_first)
        down_2_first = self.down_2(pool_1_first)
        pool_2_first = self.pool_2(down_2_first)
        down_3_first = self.down_3(pool_2_first)
        pool_3_first = self.pool_3(down_3_first)
        down_4_first = self.down_4(pool_3_first)
        pool_4_first = self.pool_4(down_4_first)
        bridge_first = self.bridge(pool_4_first)

        if secondImage is not None:
            down_1_second = self.down_1(secondImage)
            pool_1_second = self.pool_1(down_1_second)
            down_2_second = self.down_2(pool_1_second)
            pool_2_second = self.pool_2(down_2_second)
            down_3_second = self.down_3(pool_2_second)
            pool_3_second = self.pool_3(down_3_second)
            down_4_second = self.down_4(pool_3_second)
            pool_4_second = self.pool_4(down_4_second)
            bridge_second = self.bridge(pool_4_second)

            down_1_first = difference_revised(down_1_second, down_1_first, threshold=threshold_down1)
            down_2_first = difference_revised(down_2_second, down_2_first, threshold=threshold_down2)
            down_3_first = difference_revised(down_3_second, down_3_first, threshold=threshold_down3)
            down_4_first = difference_revised(down_4_second, down_4_first, threshold=threshold_down4)
            bridge_first = difference_revised(bridge_second, bridge_first, threshold=threshold_bridge)

        trans_1 = self.trans_1(bridge_first)
        concat_1 = torch.cat([trans_1, down_4_first], dim=1)
        up_1 = self.up_1(concat_1)

        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down_3_first], dim=1)
        up_2 = self.up_2(concat_2)

        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down_2_first], dim=1)
        up_3 = self.up_3(concat_3)

        trans_4 = self.trans_4(up_3)
        concat_4 = torch.cat([trans_4, down_1_first], dim=1)
        up_4 = self.up_4(concat_4)

        out = self.out(up_4)

        return out
