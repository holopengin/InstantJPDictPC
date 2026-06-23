// Generated from ONNX "assets/meiki.text.detect.v0.1.960x544.onnx" by burn-onnx
use burn::prelude::*;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::LinearLayout;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::tensor::Bytes;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;


#[derive(Module, Debug)]
pub struct Submodule1 {
    conv2d1: Conv2d,
    conv2d2: Conv2d,
    conv2d3: Conv2d,
    conv2d4: Conv2d,
    conv2d5: Conv2d,
    conv2d6: Conv2d,
    conv2d7: Conv2d,
    conv2d8: Conv2d,
    conv2d9: Conv2d,
    conv2d10: Conv2d,
    conv2d11: Conv2d,
    conv2d12: Conv2d,
    conv2d13: Conv2d,
    conv2d14: Conv2d,
    conv2d15: Conv2d,
    conv2d16: Conv2d,
    conv2d17: Conv2d,
    conv2d18: Conv2d,
    conv2d19: Conv2d,
    conv2d20: Conv2d,
    conv2d21: Conv2d,
    conv2d22: Conv2d,
    conv2d23: Conv2d,
    conv2d24: Conv2d,
    conv2d25: Conv2d,
    conv2d26: Conv2d,
    conv2d27: Conv2d,
    conv2d28: Conv2d,
    conv2d29: Conv2d,
    conv2d30: Conv2d,
    conv2d31: Conv2d,
    conv2d32: Conv2d,
    conv2d33: Conv2d,
    conv2d34: Conv2d,
    conv2d35: Conv2d,
    #[module(skip)]
    device: Device,
}
impl Submodule1 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let conv2d1 = Conv2dConfig::new([3, 32], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d2 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d3 = Conv2dConfig::new([32, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d4 = Conv2dConfig::new([32, 96], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d5 = Conv2dConfig::new([96, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d6 = Conv2dConfig::new([64, 64], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(64)
            .with_bias(true)
            .init(device);
        let conv2d7 = Conv2dConfig::new([64, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d8 = Conv2dConfig::new([192, 192], [5, 5])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d9 = Conv2dConfig::new([192, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d10 = Conv2dConfig::new([96, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d11 = Conv2dConfig::new([192, 192], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d12 = Conv2dConfig::new([192, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d13 = Conv2dConfig::new([96, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d14 = Conv2dConfig::new([192, 192], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d15 = Conv2dConfig::new([192, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d16 = Conv2dConfig::new([96, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d17 = Conv2dConfig::new([192, 192], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d18 = Conv2dConfig::new([192, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d19 = Conv2dConfig::new([96, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d20 = Conv2dConfig::new([192, 192], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(true)
            .init(device);
        let conv2d21 = Conv2dConfig::new([192, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d22 = Conv2dConfig::new([96, 96], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(96)
            .with_bias(true)
            .init(device);
        let conv2d23 = Conv2dConfig::new([96, 384], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d24 = Conv2dConfig::new([384, 96], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d25 = Conv2dConfig::new([96, 96], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(96)
            .with_bias(true)
            .init(device);
        let conv2d26 = Conv2dConfig::new([96, 576], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d27 = Conv2dConfig::new([576, 576], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(576)
            .with_bias(true)
            .init(device);
        let conv2d28 = Conv2dConfig::new([576, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d29 = Conv2dConfig::new([128, 128], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(128)
            .with_bias(true)
            .init(device);
        let conv2d30 = Conv2dConfig::new([128, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d31 = Conv2dConfig::new([512, 512], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(512)
            .with_bias(true)
            .init(device);
        let conv2d32 = Conv2dConfig::new([512, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d33 = Conv2dConfig::new([128, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d34 = Conv2dConfig::new([512, 512], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(512)
            .with_bias(true)
            .init(device);
        let conv2d35 = Conv2dConfig::new([512, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            conv2d1,
            conv2d2,
            conv2d3,
            conv2d4,
            conv2d5,
            conv2d6,
            conv2d7,
            conv2d8,
            conv2d9,
            conv2d10,
            conv2d11,
            conv2d12,
            conv2d13,
            conv2d14,
            conv2d15,
            conv2d16,
            conv2d17,
            conv2d18,
            conv2d19,
            conv2d20,
            conv2d21,
            conv2d22,
            conv2d23,
            conv2d24,
            conv2d25,
            conv2d26,
            conv2d27,
            conv2d28,
            conv2d29,
            conv2d30,
            conv2d31,
            conv2d32,
            conv2d33,
            conv2d34,
            conv2d35,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, images: Tensor<4>) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        let conv2d1_out1 = self.conv2d1.forward(images);
        let relu1_out1 = burn::tensor::activation::relu(conv2d1_out1);
        let conv2d2_out1 = self.conv2d2.forward(relu1_out1);
        let relu2_out1 = burn::tensor::activation::relu(conv2d2_out1);
        let conv2d3_out1 = self.conv2d3.forward(relu2_out1);
        let relu3_out1 = burn::tensor::activation::relu(conv2d3_out1);
        let conv2d4_out1 = self.conv2d4.forward(relu3_out1);
        let relu4_out1 = burn::tensor::activation::relu(conv2d4_out1);
        let conv2d5_out1 = self.conv2d5.forward(relu4_out1);
        let relu5_out1 = burn::tensor::activation::relu(conv2d5_out1);
        let conv2d6_out1 = self.conv2d6.forward(relu5_out1.clone());
        let conv2d7_out1 = self.conv2d7.forward(conv2d6_out1);
        let relu6_out1 = burn::tensor::activation::relu(conv2d7_out1);
        let conv2d8_out1 = self.conv2d8.forward(relu6_out1);
        let relu7_out1 = burn::tensor::activation::relu(conv2d8_out1);
        let conv2d9_out1 = self.conv2d9.forward(relu7_out1);
        let conv2d10_out1 = self.conv2d10.forward(conv2d9_out1.clone());
        let relu8_out1 = burn::tensor::activation::relu(conv2d10_out1);
        let conv2d11_out1 = self.conv2d11.forward(relu8_out1);
        let relu9_out1 = burn::tensor::activation::relu(conv2d11_out1);
        let conv2d12_out1 = self.conv2d12.forward(relu9_out1);
        let add1_out1 = conv2d12_out1.add(conv2d9_out1);
        let conv2d13_out1 = self.conv2d13.forward(add1_out1.clone());
        let relu10_out1 = burn::tensor::activation::relu(conv2d13_out1);
        let conv2d14_out1 = self.conv2d14.forward(relu10_out1);
        let relu11_out1 = burn::tensor::activation::relu(conv2d14_out1);
        let conv2d15_out1 = self.conv2d15.forward(relu11_out1);
        let add2_out1 = conv2d15_out1.add(add1_out1);
        let conv2d16_out1 = self.conv2d16.forward(add2_out1.clone());
        let relu12_out1 = burn::tensor::activation::relu(conv2d16_out1);
        let conv2d17_out1 = self.conv2d17.forward(relu12_out1);
        let relu13_out1 = burn::tensor::activation::relu(conv2d17_out1);
        let conv2d18_out1 = self.conv2d18.forward(relu13_out1);
        let add3_out1 = conv2d18_out1.add(add2_out1);
        let conv2d19_out1 = self.conv2d19.forward(add3_out1.clone());
        let relu14_out1 = burn::tensor::activation::relu(conv2d19_out1);
        let conv2d20_out1 = self.conv2d20.forward(relu14_out1);
        let relu15_out1 = burn::tensor::activation::relu(conv2d20_out1);
        let conv2d21_out1 = self.conv2d21.forward(relu15_out1);
        let add4_out1 = conv2d21_out1.add(add3_out1);
        let conv2d22_out1 = self.conv2d22.forward(add4_out1.clone());
        let conv2d23_out1 = self.conv2d23.forward(conv2d22_out1);
        let relu16_out1 = burn::tensor::activation::relu(conv2d23_out1);
        let conv2d24_out1 = self.conv2d24.forward(relu16_out1);
        let add5_out1 = conv2d24_out1.add(add4_out1);
        let conv2d25_out1 = self.conv2d25.forward(add5_out1.clone());
        let conv2d26_out1 = self.conv2d26.forward(conv2d25_out1);
        let relu17_out1 = burn::tensor::activation::relu(conv2d26_out1);
        let conv2d27_out1 = self.conv2d27.forward(relu17_out1);
        let relu18_out1 = burn::tensor::activation::relu(conv2d27_out1);
        let conv2d28_out1 = self.conv2d28.forward(relu18_out1);
        let conv2d29_out1 = self.conv2d29.forward(conv2d28_out1.clone());
        let conv2d30_out1 = self.conv2d30.forward(conv2d29_out1);
        let relu19_out1 = burn::tensor::activation::relu(conv2d30_out1);
        let conv2d31_out1 = self.conv2d31.forward(relu19_out1);
        let relu20_out1 = burn::tensor::activation::relu(conv2d31_out1);
        let conv2d32_out1 = self.conv2d32.forward(relu20_out1);
        let add6_out1 = conv2d32_out1.add(conv2d28_out1);
        let conv2d33_out1 = self.conv2d33.forward(add6_out1.clone());
        let relu21_out1 = burn::tensor::activation::relu(conv2d33_out1);
        let conv2d34_out1 = self.conv2d34.forward(relu21_out1);
        let relu22_out1 = burn::tensor::activation::relu(conv2d34_out1);
        let conv2d35_out1 = self.conv2d35.forward(relu22_out1);
        let add7_out1 = conv2d35_out1.add(add6_out1);
        (add7_out1, relu5_out1, add5_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule2 {
    conv2d36: Conv2d,
    conv2d37: Conv2d,
    conv2d38: Conv2d,
    conv2d39: Conv2d,
    conv2d40: Conv2d,
    conv2d41: Conv2d,
    conv2d42: Conv2d,
    conv2d43: Conv2d,
    conv2d44: Conv2d,
    conv2d45: Conv2d,
    conv2d46: Conv2d,
    conv2d47: Conv2d,
    conv2d48: Conv2d,
    constant373: burn::module::Param<Tensor<3>>,
    linear1: Linear,
    linear2: Linear,
    linear3: Linear,
    constant350: burn::module::Param<Tensor<1>>,
    linear4: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule2 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let conv2d36 = Conv2dConfig::new([128, 384], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d37 = Conv2dConfig::new([384, 384], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(384)
            .with_bias(true)
            .init(device);
        let conv2d38 = Conv2dConfig::new([384, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d39 = Conv2dConfig::new([128, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d40 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(512)
            .with_bias(true)
            .init(device);
        let conv2d41 = Conv2dConfig::new([512, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d42 = Conv2dConfig::new([128, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d43 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(512)
            .with_bias(true)
            .init(device);
        let conv2d44 = Conv2dConfig::new([512, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d45 = Conv2dConfig::new([128, 960], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d46 = Conv2dConfig::new([64, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d47 = Conv2dConfig::new([96, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d48 = Conv2dConfig::new([960, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let constant373: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([1, 510, 128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [1, 510, 128].into(),
        );
        let linear1 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear2 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear3 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let constant350: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.25f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let linear4 = LinearConfig::new(128, 128)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        Self {
            conv2d36,
            conv2d37,
            conv2d38,
            conv2d39,
            conv2d40,
            conv2d41,
            conv2d42,
            conv2d43,
            conv2d44,
            conv2d45,
            conv2d46,
            conv2d47,
            conv2d48,
            constant373,
            linear1,
            linear2,
            linear3,
            constant350,
            linear4,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add7_out1: Tensor<4>,
        relu5_out1: Tensor<4>,
        add5_out1: Tensor<4>,
    ) -> (Tensor<3>, Tensor<4>, Tensor<4>, [i64; 1], Tensor<1>) {
        let conv2d36_out1 = self.conv2d36.forward(add7_out1.clone());
        let relu23_out1 = burn::tensor::activation::relu(conv2d36_out1);
        let conv2d37_out1 = self.conv2d37.forward(relu23_out1);
        let relu24_out1 = burn::tensor::activation::relu(conv2d37_out1);
        let conv2d38_out1 = self.conv2d38.forward(relu24_out1);
        let add8_out1 = conv2d38_out1.add(add7_out1);
        let conv2d39_out1 = self.conv2d39.forward(add8_out1.clone());
        let relu25_out1 = burn::tensor::activation::relu(conv2d39_out1);
        let conv2d40_out1 = self.conv2d40.forward(relu25_out1);
        let relu26_out1 = burn::tensor::activation::relu(conv2d40_out1);
        let conv2d41_out1 = self.conv2d41.forward(relu26_out1);
        let add9_out1 = conv2d41_out1.add(add8_out1);
        let conv2d42_out1 = self.conv2d42.forward(add9_out1.clone());
        let relu27_out1 = burn::tensor::activation::relu(conv2d42_out1);
        let conv2d43_out1 = self.conv2d43.forward(relu27_out1);
        let relu28_out1 = burn::tensor::activation::relu(conv2d43_out1);
        let conv2d44_out1 = self.conv2d44.forward(relu28_out1);
        let add10_out1 = conv2d44_out1.add(add9_out1);
        let conv2d45_out1 = self.conv2d45.forward(add10_out1);
        let relu29_out1 = burn::tensor::activation::relu(conv2d45_out1);
        let conv2d46_out1 = self.conv2d46.forward(relu5_out1);
        let conv2d47_out1 = self.conv2d47.forward(add5_out1);
        let conv2d48_out1 = self.conv2d48.forward(relu29_out1);
        let shape1_out1: [i64; 4] = {
            let axes = &conv2d48_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice1_out1: [i64; 2] = shape1_out1[0..2].try_into().unwrap();
        let constant346_out1: [i64; 1] = [-1i64];
        let concat1_out1: [i64; 3usize] = [&slice1_out1[..], &constant346_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let reshape1_out1 = conv2d48_out1.reshape(concat1_out1);
        let transpose1_out1 = reshape1_out1.clone().permute([0, 2, 1]);
        let constant373_out1 = self.constant373.val();
        let add11_out1 = transpose1_out1.clone().add(constant373_out1);
        let transpose2_out1 = add11_out1.permute([1, 0, 2]);
        let transpose3_out1 = reshape1_out1.permute([2, 0, 1]);
        let shape2_out1: [i64; 3] = {
            let axes = &transpose2_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather1_out1 = shape2_out1[1] as i64;
        let linear1_out1 = self.linear1.forward(transpose2_out1.clone());
        let linear2_out1 = self.linear2.forward(transpose2_out1);
        let linear3_out1 = self.linear3.forward(transpose3_out1);
        let constant349_out1 = 8i64;
        let mul1_out1 = gather1_out1 * constant349_out1;
        let unsqueeze1_out1 = [mul1_out1 as i64];
        let reshape2_out1 = linear1_out1.reshape([510, -1, 16]);
        let transpose4_out1 = reshape2_out1.permute([1, 0, 2]);
        let reshape3_out1 = linear2_out1.reshape([510, -1, 16]);
        let shape3_out1: [i64; 3] = {
            let axes = &linear3_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather2_out1 = shape3_out1[0] as i64;
        let unsqueeze2_out1 = [gather2_out1 as i64];
        let constant374_out1: [i64; 1] = [16i64];
        let concat2_out1: [i64; 3usize] = [
            &unsqueeze2_out1[..],
            &unsqueeze1_out1[..],
            &constant374_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape4_out1 = linear3_out1.reshape(concat2_out1);
        let transpose5_out1 = reshape4_out1.permute([1, 0, 2]);
        let constant350_out1 = self.constant350.val();
        let mul2_out1 = transpose4_out1
            .mul((constant350_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let transpose6_out1 = reshape3_out1.permute([1, 2, 0]);
        let matmul4_out1 = mul2_out1.matmul(transpose6_out1);
        let softmax1_out1 = burn::tensor::activation::softmax(matmul4_out1, 2);
        let matmul5_out1 = softmax1_out1.matmul(transpose5_out1);
        let transpose7_out1 = matmul5_out1.permute([1, 0, 2]);
        let reshape5_out1 = transpose7_out1.reshape([-1, 128]);
        let linear4_out1 = self.linear4.forward(reshape5_out1);
        let reshape6_out1 = linear4_out1.reshape([510, -1, 128]);
        let transpose8_out1 = reshape6_out1.permute([1, 0, 2]);
        let add12_out1 = transpose1_out1.add(transpose8_out1);
        let reducemean1_out1 = { add12_out1.clone().mean_dim(2usize) };
        let sub1_out1 = add12_out1.sub(reducemean1_out1);
        (sub1_out1, conv2d47_out1, conv2d46_out1, constant346_out1, constant350_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule3 {
    constant351: burn::module::Param<Tensor<1>>,
    constant352: burn::module::Param<Tensor<1>>,
    constant68: burn::module::Param<Tensor<1>>,
    constant69: burn::module::Param<Tensor<1>>,
    linear5: Linear,
    constant353: burn::module::Param<Tensor<1>>,
    constant354: burn::module::Param<Tensor<1>>,
    constant355: burn::module::Param<Tensor<1>>,
    linear6: Linear,
    constant70: burn::module::Param<Tensor<1>>,
    constant71: burn::module::Param<Tensor<1>>,
    conv2d49: Conv2d,
    conv2d50: Conv2d,
    conv2d51: Conv2d,
    conv2d52: Conv2d,
    conv2d53: Conv2d,
    conv2d54: Conv2d,
    conv2d55: Conv2d,
    conv2d56: Conv2d,
    conv2d57: Conv2d,
    conv2d58: Conv2d,
    conv2d59: Conv2d,
    conv2d60: Conv2d,
    conv2d61: Conv2d,
    conv2d62: Conv2d,
    conv2d63: Conv2d,
    conv2d64: Conv2d,
    conv2d65: Conv2d,
    conv2d66: Conv2d,
    conv2d67: Conv2d,
    conv2d68: Conv2d,
    conv2d69: Conv2d,
    conv2d70: Conv2d,
    conv2d71: Conv2d,
    conv2d72: Conv2d,
    conv2d73: Conv2d,
    conv2d74: Conv2d,
    conv2d75: Conv2d,
    conv2d76: Conv2d,
    conv2d77: Conv2d,
    conv2d78: Conv2d,
    conv2d79: Conv2d,
    conv2d80: Conv2d,
    conv2d81: Conv2d,
    conv2d82: Conv2d,
    conv2d83: Conv2d,
    conv2d84: Conv2d,
    conv2d85: Conv2d,
    conv2d86: Conv2d,
    conv2d87: Conv2d,
    conv2d88: Conv2d,
    conv2d89: Conv2d,
    conv2d90: Conv2d,
    conv2d91: Conv2d,
    conv2d92: Conv2d,
    conv2d93: Conv2d,
    conv2d94: Conv2d,
    conv2d95: Conv2d,
    conv2d96: Conv2d,
    conv2d97: Conv2d,
    conv2d98: Conv2d,
    conv2d99: Conv2d,
    conv2d100: Conv2d,
    conv2d101: Conv2d,
    conv2d102: Conv2d,
    #[module(skip)]
    device: Device,
}
impl Submodule3 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let constant351: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([2f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant352: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.000009999999747378752f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant68: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [128].into(),
        );
        let constant69: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [128].into(),
        );
        let linear5 = LinearConfig::new(128, 512).with_bias(true).init(device);
        let constant353: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([1.4142135381698608f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant354: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([1f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let constant355: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::from_data(
                burn::tensor::TensorData::from([0.5f64]),
                (device, burn::tensor::DType::F32),
            ),
            device.clone(),
            false,
            [1].into(),
        );
        let linear6 = LinearConfig::new(512, 128).with_bias(true).init(device);
        let constant70: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [128].into(),
        );
        let constant71: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [128].into(),
        );
        let conv2d49 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d50 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d51 = Conv2dConfig::new([128, 21], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d52 = Conv2dConfig::new([21, 21], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d53 = Conv2dConfig::new([21, 21], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d54 = Conv2dConfig::new([128, 21], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d55 = Conv2dConfig::new([21, 21], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d56 = Conv2dConfig::new([21, 21], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d57 = Conv2dConfig::new([21, 21], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d58 = Conv2dConfig::new([21, 21], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d59 = Conv2dConfig::new([21, 21], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d60 = Conv2dConfig::new([21, 21], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d61 = Conv2dConfig::new([298, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d62 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d63 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d64 = Conv2dConfig::new([128, 21], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d65 = Conv2dConfig::new([21, 21], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d66 = Conv2dConfig::new([21, 21], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d67 = Conv2dConfig::new([128, 21], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d68 = Conv2dConfig::new([21, 21], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d69 = Conv2dConfig::new([21, 21], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d70 = Conv2dConfig::new([21, 21], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d71 = Conv2dConfig::new([21, 21], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d72 = Conv2dConfig::new([21, 21], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d73 = Conv2dConfig::new([21, 21], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d74 = Conv2dConfig::new([298, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d75 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d76 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(128)
            .with_bias(true)
            .init(device);
        let conv2d77 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d78 = Conv2dConfig::new([128, 21], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d79 = Conv2dConfig::new([21, 21], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d80 = Conv2dConfig::new([21, 21], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d81 = Conv2dConfig::new([128, 21], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d82 = Conv2dConfig::new([21, 21], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d83 = Conv2dConfig::new([21, 21], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d84 = Conv2dConfig::new([21, 21], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d85 = Conv2dConfig::new([21, 21], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d86 = Conv2dConfig::new([21, 21], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d87 = Conv2dConfig::new([21, 21], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d88 = Conv2dConfig::new([298, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d89 = Conv2dConfig::new([128, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d90 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(128)
            .with_bias(true)
            .init(device);
        let conv2d91 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d92 = Conv2dConfig::new([128, 21], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d93 = Conv2dConfig::new([21, 21], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d94 = Conv2dConfig::new([21, 21], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d95 = Conv2dConfig::new([128, 21], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d96 = Conv2dConfig::new([21, 21], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d97 = Conv2dConfig::new([21, 21], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d98 = Conv2dConfig::new([21, 21], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d99 = Conv2dConfig::new([21, 21], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d100 = Conv2dConfig::new([21, 21], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d101 = Conv2dConfig::new([21, 21], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d102 = Conv2dConfig::new([298, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            constant351,
            constant352,
            constant68,
            constant69,
            linear5,
            constant353,
            constant354,
            constant355,
            linear6,
            constant70,
            constant71,
            conv2d49,
            conv2d50,
            conv2d51,
            conv2d52,
            conv2d53,
            conv2d54,
            conv2d55,
            conv2d56,
            conv2d57,
            conv2d58,
            conv2d59,
            conv2d60,
            conv2d61,
            conv2d62,
            conv2d63,
            conv2d64,
            conv2d65,
            conv2d66,
            conv2d67,
            conv2d68,
            conv2d69,
            conv2d70,
            conv2d71,
            conv2d72,
            conv2d73,
            conv2d74,
            conv2d75,
            conv2d76,
            conv2d77,
            conv2d78,
            conv2d79,
            conv2d80,
            conv2d81,
            conv2d82,
            conv2d83,
            conv2d84,
            conv2d85,
            conv2d86,
            conv2d87,
            conv2d88,
            conv2d89,
            conv2d90,
            conv2d91,
            conv2d92,
            conv2d93,
            conv2d94,
            conv2d95,
            conv2d96,
            conv2d97,
            conv2d98,
            conv2d99,
            conv2d100,
            conv2d101,
            conv2d102,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        sub1_out1: Tensor<3>,
        conv2d47_out1: Tensor<4>,
        conv2d46_out1: Tensor<4>,
        constant346_out1: [i64; 1],
    ) -> (Tensor<3>, Tensor<1>, Tensor<1>, Tensor<1>, Tensor<1>) {
        let constant351_out1 = self.constant351.val();
        let pow1_out1 = sub1_out1
            .clone()
            .powf((constant351_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean2_out1 = { pow1_out1.mean_dim(2usize) };
        let constant352_out1 = self.constant352.val();
        let add13_out1 = reducemean2_out1
            .add((constant352_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt1_out1 = add13_out1.sqrt();
        let div1_out1 = sub1_out1.div(sqrt1_out1);
        let constant68_out1 = self.constant68.val();
        let mul3_out1 = div1_out1
            .mul((constant68_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant69_out1 = self.constant69.val();
        let add14_out1 = mul3_out1
            .add((constant69_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear5_out1 = self.linear5.forward(add14_out1.clone());
        let constant353_out1 = self.constant353.val();
        let div2_out1 = linear5_out1
            .clone()
            .div((constant353_out1).unsqueeze_dims(&[0isize, 1isize]));
        let erf1_out1 = div2_out1.erf();
        let constant354_out1 = self.constant354.val();
        let add15_out1 = erf1_out1
            .add((constant354_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let mul4_out1 = linear5_out1.mul(add15_out1);
        let constant355_out1 = self.constant355.val();
        let mul5_out1 = mul4_out1
            .mul((constant355_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let linear6_out1 = self.linear6.forward(mul5_out1);
        let add16_out1 = add14_out1.add(linear6_out1);
        let reducemean3_out1 = { add16_out1.clone().mean_dim(2usize) };
        let sub2_out1 = add16_out1.sub(reducemean3_out1);
        let pow2_out1 = sub2_out1
            .clone()
            .powf((constant351_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean4_out1 = { pow2_out1.mean_dim(2usize) };
        let add17_out1 = reducemean4_out1
            .add((constant352_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt2_out1 = add17_out1.sqrt();
        let div3_out1 = sub2_out1.div(sqrt2_out1);
        let constant70_out1 = self.constant70.val();
        let mul6_out1 = div3_out1
            .mul((constant70_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant71_out1 = self.constant71.val();
        let add18_out1 = mul6_out1
            .add((constant71_out1).unsqueeze_dims(&[0isize, 1isize]));
        let transpose9_out1 = add18_out1.permute([0, 2, 1]);
        let reshape7_out1 = transpose9_out1.reshape([-1, 128, 17, 30]);
        let conv2d49_out1 = self.conv2d49.forward(reshape7_out1);
        let resize1_out1 = {
            let input_dims = conv2d49_out1.clone().dims();
            let target_height = ((input_dims[2] as f64) * (2.0 as f64)) as usize;
            let target_width = ((input_dims[3] as f64) * (2.0 as f64)) as usize;
            burn::tensor::module::interpolate(
                conv2d49_out1.clone(),
                [target_height, target_width],
                burn::tensor::ops::InterpolateOptions::new(
                        burn::tensor::ops::InterpolateMode::Nearest,
                    )
                    .with_align_corners(false)
                    .with_coordinate_transformation(
                        burn::tensor::ops::CoordinateTransformMode::Asymmetric,
                    ),
            )
        };
        let concat3_out1 = burn::tensor::Tensor::cat(
            [resize1_out1, conv2d47_out1].into(),
            1,
        );
        let conv2d50_out1 = self.conv2d50.forward(concat3_out1);
        let sigmoid1_out1 = burn::tensor::activation::sigmoid(conv2d50_out1.clone());
        let mul7_out1 = conv2d50_out1.mul(sigmoid1_out1);
        let split_tensors = mul7_out1.split_with_sizes([128, 128].into(), 1);
        let [split1_out1, split1_out2] = split_tensors.try_into().unwrap();
        let conv2d51_out1 = self.conv2d51.forward(split1_out2.clone());
        let sigmoid2_out1 = burn::tensor::activation::sigmoid(conv2d51_out1.clone());
        let mul8_out1 = conv2d51_out1.mul(sigmoid2_out1);
        let conv2d52_out1 = self.conv2d52.forward(mul8_out1);
        let sigmoid3_out1 = burn::tensor::activation::sigmoid(conv2d52_out1.clone());
        let mul9_out1 = conv2d52_out1.mul(sigmoid3_out1);
        let conv2d53_out1 = self.conv2d53.forward(mul9_out1);
        let sigmoid4_out1 = burn::tensor::activation::sigmoid(conv2d53_out1.clone());
        let mul10_out1 = conv2d53_out1.mul(sigmoid4_out1);
        let conv2d54_out1 = self.conv2d54.forward(split1_out2.clone());
        let sigmoid5_out1 = burn::tensor::activation::sigmoid(conv2d54_out1.clone());
        let mul11_out1 = conv2d54_out1.mul(sigmoid5_out1);
        let add19_out1 = mul10_out1.add(mul11_out1);
        let conv2d55_out1 = self.conv2d55.forward(add19_out1);
        let sigmoid6_out1 = burn::tensor::activation::sigmoid(conv2d55_out1.clone());
        let mul12_out1 = conv2d55_out1.mul(sigmoid6_out1);
        let conv2d56_out1 = self.conv2d56.forward(mul12_out1.clone());
        let sigmoid7_out1 = burn::tensor::activation::sigmoid(conv2d56_out1.clone());
        let mul13_out1 = conv2d56_out1.mul(sigmoid7_out1);
        let conv2d57_out1 = self.conv2d57.forward(mul13_out1);
        let sigmoid8_out1 = burn::tensor::activation::sigmoid(conv2d57_out1.clone());
        let mul14_out1 = conv2d57_out1.mul(sigmoid8_out1);
        let conv2d58_out1 = self.conv2d58.forward(mul14_out1);
        let sigmoid9_out1 = burn::tensor::activation::sigmoid(conv2d58_out1.clone());
        let mul15_out1 = conv2d58_out1.mul(sigmoid9_out1);
        let conv2d59_out1 = self.conv2d59.forward(mul12_out1.clone());
        let sigmoid10_out1 = burn::tensor::activation::sigmoid(conv2d59_out1.clone());
        let mul16_out1 = conv2d59_out1.mul(sigmoid10_out1);
        let add20_out1 = mul15_out1.add(mul16_out1);
        let conv2d60_out1 = self.conv2d60.forward(add20_out1);
        let sigmoid11_out1 = burn::tensor::activation::sigmoid(conv2d60_out1.clone());
        let mul17_out1 = conv2d60_out1.mul(sigmoid11_out1);
        let concat4_out1 = burn::tensor::Tensor::cat(
            [split1_out1, split1_out2, mul12_out1, mul17_out1].into(),
            1,
        );
        let conv2d61_out1 = self.conv2d61.forward(concat4_out1);
        let sigmoid12_out1 = burn::tensor::activation::sigmoid(conv2d61_out1.clone());
        let mul18_out1 = conv2d61_out1.mul(sigmoid12_out1);
        let conv2d62_out1 = self.conv2d62.forward(mul18_out1);
        let resize2_out1 = {
            let input_dims = conv2d62_out1.clone().dims();
            let target_height = ((input_dims[2] as f64) * (2.0 as f64)) as usize;
            let target_width = ((input_dims[3] as f64) * (2.0 as f64)) as usize;
            burn::tensor::module::interpolate(
                conv2d62_out1.clone(),
                [target_height, target_width],
                burn::tensor::ops::InterpolateOptions::new(
                        burn::tensor::ops::InterpolateMode::Nearest,
                    )
                    .with_align_corners(false)
                    .with_coordinate_transformation(
                        burn::tensor::ops::CoordinateTransformMode::Asymmetric,
                    ),
            )
        };
        let concat5_out1 = burn::tensor::Tensor::cat(
            [resize2_out1, conv2d46_out1].into(),
            1,
        );
        let conv2d63_out1 = self.conv2d63.forward(concat5_out1);
        let sigmoid13_out1 = burn::tensor::activation::sigmoid(conv2d63_out1.clone());
        let mul19_out1 = conv2d63_out1.mul(sigmoid13_out1);
        let split_tensors = mul19_out1.split_with_sizes([128, 128].into(), 1);
        let [split2_out1, split2_out2] = split_tensors.try_into().unwrap();
        let conv2d64_out1 = self.conv2d64.forward(split2_out2.clone());
        let sigmoid14_out1 = burn::tensor::activation::sigmoid(conv2d64_out1.clone());
        let mul20_out1 = conv2d64_out1.mul(sigmoid14_out1);
        let conv2d65_out1 = self.conv2d65.forward(mul20_out1);
        let sigmoid15_out1 = burn::tensor::activation::sigmoid(conv2d65_out1.clone());
        let mul21_out1 = conv2d65_out1.mul(sigmoid15_out1);
        let conv2d66_out1 = self.conv2d66.forward(mul21_out1);
        let sigmoid16_out1 = burn::tensor::activation::sigmoid(conv2d66_out1.clone());
        let mul22_out1 = conv2d66_out1.mul(sigmoid16_out1);
        let conv2d67_out1 = self.conv2d67.forward(split2_out2.clone());
        let sigmoid17_out1 = burn::tensor::activation::sigmoid(conv2d67_out1.clone());
        let mul23_out1 = conv2d67_out1.mul(sigmoid17_out1);
        let add21_out1 = mul22_out1.add(mul23_out1);
        let conv2d68_out1 = self.conv2d68.forward(add21_out1);
        let sigmoid18_out1 = burn::tensor::activation::sigmoid(conv2d68_out1.clone());
        let mul24_out1 = conv2d68_out1.mul(sigmoid18_out1);
        let conv2d69_out1 = self.conv2d69.forward(mul24_out1.clone());
        let sigmoid19_out1 = burn::tensor::activation::sigmoid(conv2d69_out1.clone());
        let mul25_out1 = conv2d69_out1.mul(sigmoid19_out1);
        let conv2d70_out1 = self.conv2d70.forward(mul25_out1);
        let sigmoid20_out1 = burn::tensor::activation::sigmoid(conv2d70_out1.clone());
        let mul26_out1 = conv2d70_out1.mul(sigmoid20_out1);
        let conv2d71_out1 = self.conv2d71.forward(mul26_out1);
        let sigmoid21_out1 = burn::tensor::activation::sigmoid(conv2d71_out1.clone());
        let mul27_out1 = conv2d71_out1.mul(sigmoid21_out1);
        let conv2d72_out1 = self.conv2d72.forward(mul24_out1.clone());
        let sigmoid22_out1 = burn::tensor::activation::sigmoid(conv2d72_out1.clone());
        let mul28_out1 = conv2d72_out1.mul(sigmoid22_out1);
        let add22_out1 = mul27_out1.add(mul28_out1);
        let conv2d73_out1 = self.conv2d73.forward(add22_out1);
        let sigmoid23_out1 = burn::tensor::activation::sigmoid(conv2d73_out1.clone());
        let mul29_out1 = conv2d73_out1.mul(sigmoid23_out1);
        let concat6_out1 = burn::tensor::Tensor::cat(
            [split2_out1, split2_out2, mul24_out1, mul29_out1].into(),
            1,
        );
        let conv2d74_out1 = self.conv2d74.forward(concat6_out1);
        let sigmoid24_out1 = burn::tensor::activation::sigmoid(conv2d74_out1.clone());
        let mul30_out1 = conv2d74_out1.mul(sigmoid24_out1);
        let conv2d75_out1 = self.conv2d75.forward(mul30_out1.clone());
        let conv2d76_out1 = self.conv2d76.forward(conv2d75_out1);
        let concat7_out1 = burn::tensor::Tensor::cat(
            [conv2d76_out1, conv2d62_out1].into(),
            1,
        );
        let conv2d77_out1 = self.conv2d77.forward(concat7_out1);
        let sigmoid25_out1 = burn::tensor::activation::sigmoid(conv2d77_out1.clone());
        let mul31_out1 = conv2d77_out1.mul(sigmoid25_out1);
        let split_tensors = mul31_out1.split_with_sizes([128, 128].into(), 1);
        let [split3_out1, split3_out2] = split_tensors.try_into().unwrap();
        let conv2d78_out1 = self.conv2d78.forward(split3_out2.clone());
        let sigmoid26_out1 = burn::tensor::activation::sigmoid(conv2d78_out1.clone());
        let mul32_out1 = conv2d78_out1.mul(sigmoid26_out1);
        let conv2d79_out1 = self.conv2d79.forward(mul32_out1);
        let sigmoid27_out1 = burn::tensor::activation::sigmoid(conv2d79_out1.clone());
        let mul33_out1 = conv2d79_out1.mul(sigmoid27_out1);
        let conv2d80_out1 = self.conv2d80.forward(mul33_out1);
        let sigmoid28_out1 = burn::tensor::activation::sigmoid(conv2d80_out1.clone());
        let mul34_out1 = conv2d80_out1.mul(sigmoid28_out1);
        let conv2d81_out1 = self.conv2d81.forward(split3_out2.clone());
        let sigmoid29_out1 = burn::tensor::activation::sigmoid(conv2d81_out1.clone());
        let mul35_out1 = conv2d81_out1.mul(sigmoid29_out1);
        let add23_out1 = mul34_out1.add(mul35_out1);
        let conv2d82_out1 = self.conv2d82.forward(add23_out1);
        let sigmoid30_out1 = burn::tensor::activation::sigmoid(conv2d82_out1.clone());
        let mul36_out1 = conv2d82_out1.mul(sigmoid30_out1);
        let conv2d83_out1 = self.conv2d83.forward(mul36_out1.clone());
        let sigmoid31_out1 = burn::tensor::activation::sigmoid(conv2d83_out1.clone());
        let mul37_out1 = conv2d83_out1.mul(sigmoid31_out1);
        let conv2d84_out1 = self.conv2d84.forward(mul37_out1);
        let sigmoid32_out1 = burn::tensor::activation::sigmoid(conv2d84_out1.clone());
        let mul38_out1 = conv2d84_out1.mul(sigmoid32_out1);
        let conv2d85_out1 = self.conv2d85.forward(mul38_out1);
        let sigmoid33_out1 = burn::tensor::activation::sigmoid(conv2d85_out1.clone());
        let mul39_out1 = conv2d85_out1.mul(sigmoid33_out1);
        let conv2d86_out1 = self.conv2d86.forward(mul36_out1.clone());
        let sigmoid34_out1 = burn::tensor::activation::sigmoid(conv2d86_out1.clone());
        let mul40_out1 = conv2d86_out1.mul(sigmoid34_out1);
        let add24_out1 = mul39_out1.add(mul40_out1);
        let conv2d87_out1 = self.conv2d87.forward(add24_out1);
        let sigmoid35_out1 = burn::tensor::activation::sigmoid(conv2d87_out1.clone());
        let mul41_out1 = conv2d87_out1.mul(sigmoid35_out1);
        let concat8_out1 = burn::tensor::Tensor::cat(
            [split3_out1, split3_out2, mul36_out1, mul41_out1].into(),
            1,
        );
        let conv2d88_out1 = self.conv2d88.forward(concat8_out1);
        let sigmoid36_out1 = burn::tensor::activation::sigmoid(conv2d88_out1.clone());
        let mul42_out1 = conv2d88_out1.mul(sigmoid36_out1);
        let conv2d89_out1 = self.conv2d89.forward(mul42_out1.clone());
        let conv2d90_out1 = self.conv2d90.forward(conv2d89_out1);
        let concat9_out1 = burn::tensor::Tensor::cat(
            [conv2d90_out1, conv2d49_out1].into(),
            1,
        );
        let conv2d91_out1 = self.conv2d91.forward(concat9_out1);
        let sigmoid37_out1 = burn::tensor::activation::sigmoid(conv2d91_out1.clone());
        let mul43_out1 = conv2d91_out1.mul(sigmoid37_out1);
        let split_tensors = mul43_out1.split_with_sizes([128, 128].into(), 1);
        let [split4_out1, split4_out2] = split_tensors.try_into().unwrap();
        let conv2d92_out1 = self.conv2d92.forward(split4_out2.clone());
        let sigmoid38_out1 = burn::tensor::activation::sigmoid(conv2d92_out1.clone());
        let mul44_out1 = conv2d92_out1.mul(sigmoid38_out1);
        let conv2d93_out1 = self.conv2d93.forward(mul44_out1);
        let sigmoid39_out1 = burn::tensor::activation::sigmoid(conv2d93_out1.clone());
        let mul45_out1 = conv2d93_out1.mul(sigmoid39_out1);
        let conv2d94_out1 = self.conv2d94.forward(mul45_out1);
        let sigmoid40_out1 = burn::tensor::activation::sigmoid(conv2d94_out1.clone());
        let mul46_out1 = conv2d94_out1.mul(sigmoid40_out1);
        let conv2d95_out1 = self.conv2d95.forward(split4_out2.clone());
        let sigmoid41_out1 = burn::tensor::activation::sigmoid(conv2d95_out1.clone());
        let mul47_out1 = conv2d95_out1.mul(sigmoid41_out1);
        let add25_out1 = mul46_out1.add(mul47_out1);
        let conv2d96_out1 = self.conv2d96.forward(add25_out1);
        let sigmoid42_out1 = burn::tensor::activation::sigmoid(conv2d96_out1.clone());
        let mul48_out1 = conv2d96_out1.mul(sigmoid42_out1);
        let conv2d97_out1 = self.conv2d97.forward(mul48_out1.clone());
        let sigmoid43_out1 = burn::tensor::activation::sigmoid(conv2d97_out1.clone());
        let mul49_out1 = conv2d97_out1.mul(sigmoid43_out1);
        let conv2d98_out1 = self.conv2d98.forward(mul49_out1);
        let sigmoid44_out1 = burn::tensor::activation::sigmoid(conv2d98_out1.clone());
        let mul50_out1 = conv2d98_out1.mul(sigmoid44_out1);
        let conv2d99_out1 = self.conv2d99.forward(mul50_out1);
        let sigmoid45_out1 = burn::tensor::activation::sigmoid(conv2d99_out1.clone());
        let mul51_out1 = conv2d99_out1.mul(sigmoid45_out1);
        let conv2d100_out1 = self.conv2d100.forward(mul48_out1.clone());
        let sigmoid46_out1 = burn::tensor::activation::sigmoid(conv2d100_out1.clone());
        let mul52_out1 = conv2d100_out1.mul(sigmoid46_out1);
        let add26_out1 = mul51_out1.add(mul52_out1);
        let conv2d101_out1 = self.conv2d101.forward(add26_out1);
        let sigmoid47_out1 = burn::tensor::activation::sigmoid(conv2d101_out1.clone());
        let mul53_out1 = conv2d101_out1.mul(sigmoid47_out1);
        let concat10_out1 = burn::tensor::Tensor::cat(
            [split4_out1, split4_out2, mul48_out1, mul53_out1].into(),
            1,
        );
        let conv2d102_out1 = self.conv2d102.forward(concat10_out1);
        let sigmoid48_out1 = burn::tensor::activation::sigmoid(conv2d102_out1.clone());
        let mul54_out1 = conv2d102_out1.mul(sigmoid48_out1);
        let shape4_out1: [i64; 4] = {
            let axes = &mul30_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice2_out1: [i64; 2] = shape4_out1[0..2].try_into().unwrap();
        let concat11_out1: [i64; 3usize] = [&slice2_out1[..], &constant346_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let reshape8_out1 = mul30_out1.reshape(concat11_out1);
        let transpose10_out1 = reshape8_out1.permute([0, 2, 1]);
        let shape5_out1: [i64; 4] = {
            let axes = &mul42_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice3_out1: [i64; 2] = shape5_out1[0..2].try_into().unwrap();
        let concat12_out1: [i64; 3usize] = [&slice3_out1[..], &constant346_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let reshape9_out1 = mul42_out1.reshape(concat12_out1);
        let transpose11_out1 = reshape9_out1.permute([0, 2, 1]);
        let shape6_out1: [i64; 4] = {
            let axes = &mul54_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let slice4_out1: [i64; 2] = shape6_out1[0..2].try_into().unwrap();
        let concat13_out1: [i64; 3usize] = [&slice4_out1[..], &constant346_out1[..]]
            .concat()
            .try_into()
            .unwrap();
        let reshape10_out1 = mul54_out1.reshape(concat13_out1);
        let transpose12_out1 = reshape10_out1.permute([0, 2, 1]);
        let concat14_out1 = burn::tensor::Tensor::cat(
            [transpose10_out1, transpose11_out1, transpose12_out1].into(),
            1,
        );
        (
            concat14_out1,
            constant351_out1,
            constant352_out1,
            constant355_out1,
            constant354_out1,
        )
    }
}
#[derive(Module, Debug)]
pub struct Submodule4 {
    constant380: burn::module::Param<Tensor<3>>,
    constant381: burn::module::Param<Tensor<3>>,
    linear7: Linear,
    constant45: burn::module::Param<Tensor<1>>,
    constant46: burn::module::Param<Tensor<1>>,
    linear8: Linear,
    concat16: burn::module::Param<Tensor<1, Int>>,
    linear9: Linear,
    linear10: Linear,
    linear11: Linear,
    linear12: Linear,
    linear13: Linear,
    linear14: Linear,
    linear15: Linear,
    linear16: Linear,
    linear17: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule4 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let constant380: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([1, 10710, 4], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [1, 10710, 4].into(),
        );
        let constant381: burn::module::Param<Tensor<3>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                3,
            >::zeros([1, 10710, 1], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [1, 10710, 1].into(),
        );
        let linear7 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let constant45: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [128].into(),
        );
        let constant46: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [128].into(),
        );
        let linear8 = LinearConfig::new(128, 1).with_bias(true).init(device);
        let concat16: burn::module::Param<Tensor<1, Int>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
                Int,
            >::zeros([3], (device, burn::tensor::DType::I64)),
            device.clone(),
            false,
            [3].into(),
        );
        let linear9 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear10 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear11 = LinearConfig::new(128, 4).with_bias(true).init(device);
        let linear12 = LinearConfig::new(4, 256).with_bias(true).init(device);
        let linear13 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let linear14 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear15 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear16 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear17 = LinearConfig::new(128, 128)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        Self {
            constant380,
            constant381,
            linear7,
            constant45,
            constant46,
            linear8,
            concat16,
            linear9,
            linear10,
            linear11,
            linear12,
            linear13,
            linear14,
            linear15,
            linear16,
            linear17,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        concat14_out1: Tensor<3>,
        constant351_out1: Tensor<1>,
        constant352_out1: Tensor<1>,
        constant346_out1: [i64; 1],
        constant350_out1: Tensor<1>,
    ) -> (Tensor<3>, Tensor<3>, Tensor<3>, Tensor<4>, Tensor<4>, Tensor<4>) {
        let shape7_out1: [i64; 3] = {
            let axes = &concat14_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather3_out1 = shape7_out1[0] as i64;
        let unsqueeze3_out1 = [gather3_out1 as i64];
        let constant348_out1: [i64; 1] = [1i64];
        let concat15_out1: [i64; 3usize] = [
            &unsqueeze3_out1[..],
            &constant348_out1[..],
            &constant348_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let constant380_out1 = self.constant380.val();
        let tile1_out1 = {
            let __repeats: alloc::vec::Vec<usize> = concat15_out1
                .iter()
                .map(|&v| v as usize)
                .collect();
            constant380_out1.repeat(&__repeats)
        };
        let constant381_out1 = self.constant381.val();
        let mul55_out1 = constant381_out1.mul(concat14_out1.clone());
        let linear7_out1 = self.linear7.forward(mul55_out1);
        let reducemean5_out1 = { linear7_out1.clone().mean_dim(2usize) };
        let sub3_out1 = linear7_out1.sub(reducemean5_out1);
        let pow3_out1 = sub3_out1
            .clone()
            .powf((constant351_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean6_out1 = { pow3_out1.mean_dim(2usize) };
        let add27_out1 = reducemean6_out1
            .add((constant352_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt3_out1 = add27_out1.sqrt();
        let div4_out1 = sub3_out1.div(sqrt3_out1);
        let constant45_out1 = self.constant45.val();
        let mul56_out1 = div4_out1
            .mul((constant45_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant46_out1 = self.constant46.val();
        let add28_out1 = mul56_out1
            .add((constant46_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear8_out1 = self.linear8.forward(add28_out1.clone());
        let reducemax1_out1 = {
            linear8_out1.max_dim(2usize).squeeze_dims::<2usize>(&[2])
        };
        let (topk1_out1, __topk_indices_raw) = reducemax1_out1.topk_with_indices(64, 1);
        let topk1_out2 = __topk_indices_raw.cast(burn::tensor::DType::I64);
        let unsqueeze4_out1: Tensor<3, Int> = topk1_out2.unsqueeze_dims::<3>(&[-1]);
        let concat16_out1 = self.concat16.val();
        let tile2_out1 = unsqueeze4_out1.clone().repeat(&[1, 1, 4]);
        let gatherelements1_out1 = tile1_out1.gather(1, tile2_out1);
        let tile3_out1 = unsqueeze4_out1.repeat(&[1, 1, 128]);
        let gatherelements2_out1 = add28_out1.gather(1, tile3_out1);
        let linear9_out1 = self.linear9.forward(gatherelements2_out1.clone());
        let relu30_out1 = burn::tensor::activation::relu(linear9_out1);
        let linear10_out1 = self.linear10.forward(relu30_out1);
        let relu31_out1 = burn::tensor::activation::relu(linear10_out1);
        let linear11_out1 = self.linear11.forward(relu31_out1);
        let add29_out1 = linear11_out1.add(gatherelements1_out1);
        let gather5_out1 = shape7_out1[1] as i64;
        let unsqueeze6_out1 = [gather5_out1 as i64];
        let constant360_out1: [i64; 1] = [8i64];
        let concat17_out1: [i64; 4usize] = [
            &unsqueeze3_out1[..],
            &unsqueeze6_out1[..],
            &constant360_out1[..],
            &constant346_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape11_out1 = concat14_out1.reshape(concat17_out1);
        let transpose13_out1 = reshape11_out1.permute([0, 2, 3, 1]);
        let slice5_out1 = transpose13_out1.clone().slice(s![.., .., .., 0..8160]);
        let slice6_out1 = transpose13_out1.clone().slice(s![.., .., .., 8160..10200]);
        let slice7_out1 = transpose13_out1.slice(s![.., .., .., 10200..10710]);
        let sigmoid49_out1 = burn::tensor::activation::sigmoid(add29_out1);
        let linear12_out1 = self.linear12.forward(sigmoid49_out1.clone());
        let relu32_out1 = burn::tensor::activation::relu(linear12_out1);
        let linear13_out1 = self.linear13.forward(relu32_out1);
        let clip1_out1 = {
            let __clip_min = -10f64;
            let __clip_max = 10f64;
            linear13_out1.clamp(__clip_min, __clip_max)
        };
        let add30_out1 = gatherelements2_out1.clone().add(clip1_out1.clone());
        let transpose14_out1 = add30_out1.permute([1, 0, 2]);
        let transpose15_out1 = gatherelements2_out1.clone().permute([1, 0, 2]);
        let linear14_out1 = self.linear14.forward(transpose14_out1.clone());
        let linear15_out1 = self.linear15.forward(transpose14_out1);
        let linear16_out1 = self.linear16.forward(transpose15_out1);
        let reshape12_out1 = linear14_out1.reshape([64, -1, 16]);
        let transpose16_out1 = reshape12_out1.permute([1, 0, 2]);
        let reshape13_out1 = linear15_out1.reshape([64, -1, 16]);
        let reshape14_out1 = linear16_out1.reshape([64, -1, 16]);
        let transpose17_out1 = reshape14_out1.permute([1, 0, 2]);
        let mul57_out1 = transpose16_out1
            .mul((constant350_out1).unsqueeze_dims(&[0isize, 1isize]));
        let transpose18_out1 = reshape13_out1.permute([1, 2, 0]);
        let matmul18_out1 = mul57_out1.matmul(transpose18_out1);
        let softmax2_out1 = burn::tensor::activation::softmax(matmul18_out1, 2);
        let matmul19_out1 = softmax2_out1.matmul(transpose17_out1);
        let transpose19_out1 = matmul19_out1.permute([1, 0, 2]);
        let reshape15_out1 = transpose19_out1.reshape([-1, 128]);
        let linear17_out1 = self.linear17.forward(reshape15_out1);
        let reshape16_out1 = linear17_out1.reshape([64, -1, 128]);
        let transpose20_out1 = reshape16_out1.permute([1, 0, 2]);
        let add31_out1 = gatherelements2_out1.add(transpose20_out1);
        (add31_out1, clip1_out1, sigmoid49_out1, slice5_out1, slice6_out1, slice7_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule5 {
    constant3: burn::module::Param<Tensor<1>>,
    constant4: burn::module::Param<Tensor<1>>,
    linear18: Linear,
    linear19: Linear,
    constant299: burn::module::Param<Tensor<2>>,
    linear20: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule5 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let constant3: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [128].into(),
        );
        let constant4: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [128].into(),
        );
        let linear18 = LinearConfig::new(128, 288).with_bias(true).init(device);
        let linear19 = LinearConfig::new(128, 144).with_bias(true).init(device);
        let constant299: burn::module::Param<Tensor<2>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                2,
            >::zeros([18, 1], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [18, 1].into(),
        );
        let linear20 = LinearConfig::new(256, 256).with_bias(true).init(device);
        Self {
            constant3,
            constant4,
            linear18,
            linear19,
            constant299,
            linear20,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add31_out1: Tensor<3>,
        constant351_out1: Tensor<1>,
        constant352_out1: Tensor<1>,
        clip1_out1: Tensor<3>,
        sigmoid49_out1: Tensor<3>,
        constant355_out1: Tensor<1>,
        slice5_out1: Tensor<4>,
        constant354_out1: Tensor<1>,
        slice6_out1: Tensor<4>,
        slice7_out1: Tensor<4>,
    ) -> (Tensor<3>, [i64; 1], Tensor<2>, [i64; 3]) {
        let reducemean7_out1 = { add31_out1.clone().mean_dim(2usize) };
        let sub4_out1 = add31_out1.sub(reducemean7_out1);
        let pow4_out1 = sub4_out1
            .clone()
            .powf((constant351_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean8_out1 = { pow4_out1.mean_dim(2usize) };
        let add32_out1 = reducemean8_out1
            .add((constant352_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt4_out1 = add32_out1.sqrt();
        let div5_out1 = sub4_out1.div(sqrt4_out1);
        let constant3_out1 = self.constant3.val();
        let mul58_out1 = div5_out1
            .mul((constant3_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant4_out1 = self.constant4.val();
        let add33_out1 = mul58_out1
            .add((constant4_out1).unsqueeze_dims(&[0isize, 1isize]));
        let add34_out1 = add33_out1.clone().add(clip1_out1);
        let linear18_out1 = self.linear18.forward(add34_out1.clone());
        let reshape17_out1 = linear18_out1.reshape([-1, 64, 8, 18, 2]);
        let linear19_out1 = self.linear19.forward(add34_out1);
        let reshape18_out1 = linear19_out1.reshape([-1, 64, 8, 18]);
        let softmax3_out1 = burn::tensor::activation::softmax(reshape18_out1, 3);
        let constant299_out1 = self.constant299.val();
        let mul59_out1 = reshape17_out1
            .mul((constant299_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let unsqueeze7_out1: Tensor<5> = sigmoid49_out1.unsqueeze_dims::<5>(&[2, 3]);
        let slice8_out1 = unsqueeze7_out1.clone().slice(s![.., .., .., .., 2..]);
        let mul60_out1 = mul59_out1.mul(slice8_out1);
        let mul61_out1 = mul60_out1
            .mul((constant355_out1).unsqueeze_dims(&[0isize, 1isize, 2isize, 3isize]));
        let slice9_out1 = unsqueeze7_out1.slice(s![.., .., .., .., 0..2]);
        let add35_out1 = slice9_out1.add(mul61_out1);
        let shape9_out1: [i64; 4] = {
            let axes = &slice5_out1.clone().dims()[0..4];
            let mut output = [0i64; 4];
            for i in 0..4 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather6_out1 = shape9_out1[0] as i64;
        let gather7_out1 = shape9_out1[1] as i64;
        let gather8_out1 = shape9_out1[2] as i64;
        let mul62_out1 = add35_out1
            .mul(
                (constant351_out1.clone())
                    .unsqueeze_dims(&[0isize, 1isize, 2isize, 3isize]),
            );
        let sub5_out1 = mul62_out1
            .sub((constant354_out1).unsqueeze_dims(&[0isize, 1isize, 2isize, 3isize]));
        let transpose21_out1 = sub5_out1.permute([0, 2, 1, 3, 4]);
        let reshape19_out1 = transpose21_out1.reshape([-1, 64, 18, 2]);
        let split_tensors = reshape19_out1.split_with_sizes([6, 6, 6].into(), 2);
        let [split5_out1, split5_out2, split5_out3] = split_tensors.try_into().unwrap();
        let gridsample1_out1 = slice5_out1
            .grid_sample_2d(
                split5_out1,
                burn::tensor::ops::GridSampleOptions::new(
                        burn::tensor::ops::InterpolateMode::Bilinear,
                    )
                    .with_padding_mode(burn::tensor::ops::GridSamplePaddingMode::Zeros)
                    .with_align_corners(false),
            );
        let gridsample2_out1 = slice6_out1
            .grid_sample_2d(
                split5_out2,
                burn::tensor::ops::GridSampleOptions::new(
                        burn::tensor::ops::InterpolateMode::Bilinear,
                    )
                    .with_padding_mode(burn::tensor::ops::GridSamplePaddingMode::Zeros)
                    .with_align_corners(false),
            );
        let gridsample3_out1 = slice7_out1
            .grid_sample_2d(
                split5_out3,
                burn::tensor::ops::GridSampleOptions::new(
                        burn::tensor::ops::InterpolateMode::Bilinear,
                    )
                    .with_padding_mode(burn::tensor::ops::GridSamplePaddingMode::Zeros)
                    .with_align_corners(false),
            );
        let transpose22_out1 = softmax3_out1.permute([0, 2, 1, 3]);
        let reshape23_out1 = transpose22_out1.reshape([-1, 1, 64, 18]);
        let concat21_out1 = burn::tensor::Tensor::cat(
            [gridsample1_out1, gridsample2_out1, gridsample3_out1].into(),
            3,
        );
        let mul64_out1 = concat21_out1.mul(reshape23_out1);
        let reducesum1_out1 = {
            mul64_out1.sum_dim(3usize).squeeze_dims::<3usize>(&[3])
        };
        let mul65_out1 = gather7_out1 * gather8_out1;
        let unsqueeze10_out1 = [gather6_out1 as i64];
        let unsqueeze11_out1 = [mul65_out1 as i64];
        let constant359_out1: [i64; 1] = [64i64];
        let concat22_out1: [i64; 3usize] = [
            &unsqueeze10_out1[..],
            &unsqueeze11_out1[..],
            &constant359_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape24_out1 = reducesum1_out1.reshape(concat22_out1);
        let transpose23_out1 = reshape24_out1.permute([0, 2, 1]);
        let concat23_out1 = burn::tensor::Tensor::cat(
            [add33_out1.clone(), transpose23_out1.clone()].into(),
            2,
        );
        let linear20_out1 = self.linear20.forward(concat23_out1);
        let sigmoid50_out1 = burn::tensor::activation::sigmoid(linear20_out1);
        let slice10_out1 = sigmoid50_out1.clone().slice(s![.., .., 0..128]);
        let slice11_out1 = sigmoid50_out1.slice(s![.., .., 128..256]);
        let mul66_out1 = slice10_out1.mul(add33_out1);
        let mul67_out1 = slice11_out1.mul(transpose23_out1);
        let add36_out1 = mul66_out1.add(mul67_out1);
        let reducemean9_out1 = { add36_out1.clone().mean_dim(2usize) };
        let sub6_out1 = add36_out1.sub(reducemean9_out1);
        let pow5_out1 = sub6_out1
            .clone()
            .powf((constant351_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean10_out1 = { pow5_out1.mean_dim(2usize) };
        let add37_out1 = reducemean10_out1
            .add((constant352_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt5_out1 = add37_out1.sqrt();
        let div6_out1 = sub6_out1.div(sqrt5_out1);
        (div6_out1, constant359_out1, constant299_out1, concat22_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule6 {
    constant8: burn::module::Param<Tensor<1>>,
    constant9: burn::module::Param<Tensor<1>>,
    linear21: Linear,
    linear22: Linear,
    constant12: burn::module::Param<Tensor<1>>,
    constant13: burn::module::Param<Tensor<1>>,
    linear23: Linear,
    linear24: Linear,
    linear25: Linear,
    linear26: Linear,
    linear27: Linear,
    linear28: Linear,
    constant369: burn::module::Param<Tensor<1>>,
    constant391: burn::module::Param<Tensor<1>>,
    constant390: burn::module::Param<Tensor<1>>,
    linear29: Linear,
    linear30: Linear,
    linear31: Linear,
    linear32: Linear,
    linear33: Linear,
    linear34: Linear,
    constant16: burn::module::Param<Tensor<1>>,
    constant17: burn::module::Param<Tensor<1>>,
    linear35: Linear,
    linear36: Linear,
    linear37: Linear,
    constant21: burn::module::Param<Tensor<1>>,
    constant22: burn::module::Param<Tensor<1>>,
    linear38: Linear,
    linear39: Linear,
    constant25: burn::module::Param<Tensor<1>>,
    constant26: burn::module::Param<Tensor<1>>,
    linear40: Linear,
    linear41: Linear,
    linear42: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule6 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let constant8: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [128].into(),
        );
        let constant9: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [128].into(),
        );
        let linear21 = LinearConfig::new(128, 512).with_bias(true).init(device);
        let linear22 = LinearConfig::new(512, 128).with_bias(true).init(device);
        let constant12: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [128].into(),
        );
        let constant13: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [128].into(),
        );
        let linear23 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear24 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear25 = LinearConfig::new(128, 4).with_bias(true).init(device);
        let linear26 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear27 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear28 = LinearConfig::new(128, 132).with_bias(true).init(device);
        let constant369: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([33], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [33].into(),
        );
        let constant391: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([1], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [1].into(),
        );
        let constant390: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([1], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [1].into(),
        );
        let linear29 = LinearConfig::new(4, 256).with_bias(true).init(device);
        let linear30 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let linear31 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear32 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear33 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear34 = LinearConfig::new(128, 128)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let constant16: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [128].into(),
        );
        let constant17: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [128].into(),
        );
        let linear35 = LinearConfig::new(128, 288).with_bias(true).init(device);
        let linear36 = LinearConfig::new(128, 144).with_bias(true).init(device);
        let linear37 = LinearConfig::new(256, 256).with_bias(true).init(device);
        let constant21: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [128].into(),
        );
        let constant22: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [128].into(),
        );
        let linear38 = LinearConfig::new(128, 512).with_bias(true).init(device);
        let linear39 = LinearConfig::new(512, 128).with_bias(true).init(device);
        let constant25: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [128].into(),
        );
        let constant26: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [128].into(),
        );
        let linear40 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear41 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear42 = LinearConfig::new(128, 132).with_bias(true).init(device);
        Self {
            constant8,
            constant9,
            linear21,
            linear22,
            constant12,
            constant13,
            linear23,
            linear24,
            linear25,
            linear26,
            linear27,
            linear28,
            constant369,
            constant391,
            constant390,
            linear29,
            linear30,
            linear31,
            linear32,
            linear33,
            linear34,
            constant16,
            constant17,
            linear35,
            linear36,
            linear37,
            constant21,
            constant22,
            linear38,
            linear39,
            constant25,
            constant26,
            linear40,
            linear41,
            linear42,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        div6_out1: Tensor<3>,
        constant351_out1: Tensor<1>,
        constant352_out1: Tensor<1>,
        sigmoid49_out1: Tensor<3>,
        constant354_out1: Tensor<1>,
        constant359_out1: [i64; 1],
        constant346_out1: [i64; 1],
        constant350_out1: Tensor<1>,
        constant299_out1: Tensor<2>,
        constant355_out1: Tensor<1>,
        slice5_out1: Tensor<4>,
        slice6_out1: Tensor<4>,
        slice7_out1: Tensor<4>,
        concat22_out1: [i64; 3],
    ) -> (
        Tensor<3>,
        Tensor<1>,
        Tensor<1>,
        Tensor<2>,
        Tensor<2>,
        Tensor<2>,
        Tensor<2>,
        Tensor<3>,
    ) {
        let constant8_out1 = self.constant8.val();
        let mul68_out1 = div6_out1
            .mul((constant8_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant9_out1 = self.constant9.val();
        let add38_out1 = mul68_out1
            .add((constant9_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear21_out1 = self.linear21.forward(add38_out1.clone());
        let relu33_out1 = burn::tensor::activation::relu(linear21_out1);
        let linear22_out1 = self.linear22.forward(relu33_out1);
        let add39_out1 = add38_out1.add(linear22_out1);
        let clip2_out1 = {
            let __clip_min = -65504f64;
            let __clip_max = 65504f64;
            add39_out1.clamp(__clip_min, __clip_max)
        };
        let reducemean11_out1 = { clip2_out1.clone().mean_dim(2usize) };
        let sub7_out1 = clip2_out1.sub(reducemean11_out1);
        let pow6_out1 = sub7_out1
            .clone()
            .powf((constant351_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean12_out1 = { pow6_out1.mean_dim(2usize) };
        let add40_out1 = reducemean12_out1
            .add((constant352_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt6_out1 = add40_out1.sqrt();
        let div7_out1 = sub7_out1.div(sqrt6_out1);
        let constant12_out1 = self.constant12.val();
        let mul69_out1 = div7_out1
            .mul((constant12_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant13_out1 = self.constant13.val();
        let add41_out1 = mul69_out1
            .add((constant13_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear23_out1 = self.linear23.forward(add41_out1.clone());
        let relu34_out1 = burn::tensor::activation::relu(linear23_out1);
        let linear24_out1 = self.linear24.forward(relu34_out1);
        let relu35_out1 = burn::tensor::activation::relu(linear24_out1);
        let linear25_out1 = self.linear25.forward(relu35_out1);
        let clip3_out1 = {
            let __clip_min = 0f64;
            let __clip_max = 1f64;
            sigmoid49_out1.clamp(__clip_min, __clip_max)
        };
        let clip4_out1 = {
            let __clip_min = 0.000009999999747378752f64;
            clip3_out1.clone().clamp_min(__clip_min)
        };
        let sub8_out1 = (constant354_out1.clone())
            .unsqueeze_dims(&[0isize, 1isize])
            .sub(clip3_out1);
        let clip5_out1 = {
            let __clip_min = 0.000009999999747378752f64;
            sub8_out1.clamp_min(__clip_min)
        };
        let div8_out1 = clip4_out1.div(clip5_out1);
        let log1_out1 = div8_out1.log();
        let add42_out1 = linear25_out1.add(log1_out1);
        let sigmoid51_out1 = burn::tensor::activation::sigmoid(add42_out1);
        let linear26_out1 = self.linear26.forward(add41_out1.clone());
        let relu36_out1 = burn::tensor::activation::relu(linear26_out1);
        let linear27_out1 = self.linear27.forward(relu36_out1);
        let relu37_out1 = burn::tensor::activation::relu(linear27_out1);
        let linear28_out1 = self.linear28.forward(relu37_out1);
        let shape10_out1: [i64; 3] = {
            let axes = &linear28_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather9_out1 = shape10_out1[0] as i64;
        let reshape25_out1 = linear28_out1.clone().reshape([-1, 33]);
        let softmax4_out1 = burn::tensor::activation::softmax(reshape25_out1, 1);
        let constant369_out1 = self.constant369.val();
        let matmul31_out1 = softmax4_out1
            .matmul(constant369_out1.clone().unsqueeze_dims(&[-1isize]))
            .squeeze_dim::<1usize>(1usize);
        let unsqueeze12_out1 = [gather9_out1 as i64];
        let concat24_out1: [i64; 3usize] = [
            &unsqueeze12_out1[..],
            &constant359_out1[..],
            &constant346_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape26_out1 = matmul31_out1.reshape(concat24_out1);
        let gather10_out1 = {
            let sliced = sigmoid51_out1.clone().slice(s![.., .., 0]);
            sliced.squeeze_dim::<2usize>(2)
        };
        let gather11_out1 = {
            let sliced = reshape26_out1.clone().slice(s![.., .., 0]);
            sliced.squeeze_dim::<2usize>(2)
        };
        let constant391_out1 = self.constant391.val();
        let add43_out1 = (constant391_out1.clone())
            .unsqueeze_dims(&[0isize])
            .add(gather11_out1);
        let gather12_out1 = {
            let sliced = sigmoid51_out1.clone().slice(s![.., .., 2]);
            sliced.squeeze_dim::<2usize>(2)
        };
        let constant390_out1 = self.constant390.val();
        let div9_out1 = gather12_out1
            .div((constant390_out1.clone()).unsqueeze_dims(&[0isize]));
        let mul70_out1 = add43_out1.mul(div9_out1.clone());
        let sub9_out1 = gather10_out1.clone().sub(mul70_out1);
        let gather13_out1 = {
            let sliced = sigmoid51_out1.clone().slice(s![.., .., 1]);
            sliced.squeeze_dim::<2usize>(2)
        };
        let gather14_out1 = {
            let sliced = reshape26_out1.clone().slice(s![.., .., 1]);
            sliced.squeeze_dim::<2usize>(2)
        };
        let add44_out1 = (constant391_out1.clone())
            .unsqueeze_dims(&[0isize])
            .add(gather14_out1);
        let gather15_out1 = {
            let sliced = sigmoid51_out1.slice(s![.., .., 3]);
            sliced.squeeze_dim::<2usize>(2)
        };
        let div10_out1 = gather15_out1.div((constant390_out1).unsqueeze_dims(&[0isize]));
        let mul71_out1 = add44_out1.mul(div10_out1.clone());
        let sub10_out1 = gather13_out1.clone().sub(mul71_out1);
        let gather16_out1 = {
            let sliced = reshape26_out1.clone().slice(s![.., .., 2]);
            sliced.squeeze_dim::<2usize>(2)
        };
        let add45_out1 = (constant391_out1.clone())
            .unsqueeze_dims(&[0isize])
            .add(gather16_out1);
        let mul72_out1 = add45_out1.mul(div9_out1.clone());
        let add46_out1 = gather10_out1.clone().add(mul72_out1);
        let gather17_out1 = {
            let sliced = reshape26_out1.slice(s![.., .., 3]);
            sliced.squeeze_dim::<2usize>(2)
        };
        let add47_out1 = (constant391_out1.clone())
            .unsqueeze_dims(&[0isize])
            .add(gather17_out1);
        let mul73_out1 = add47_out1.mul(div10_out1.clone());
        let add48_out1 = gather13_out1.clone().add(mul73_out1);
        let unsqueeze13_out1: Tensor<3> = sub9_out1.unsqueeze_dims::<3>(&[-1]);
        let unsqueeze14_out1: Tensor<3> = sub10_out1.unsqueeze_dims::<3>(&[-1]);
        let unsqueeze15_out1: Tensor<3> = add46_out1.unsqueeze_dims::<3>(&[-1]);
        let unsqueeze16_out1: Tensor<3> = add48_out1.unsqueeze_dims::<3>(&[-1]);
        let concat25_out1 = burn::tensor::Tensor::cat(
            [unsqueeze13_out1, unsqueeze14_out1, unsqueeze15_out1, unsqueeze16_out1]
                .into(),
            2,
        );
        let split_tensors = concat25_out1.split_with_sizes([1, 1, 1, 1].into(), 2);
        let [split6_out1, split6_out2, split6_out3, split6_out4] = split_tensors
            .try_into()
            .unwrap();
        let squeeze1_out1 = split6_out1.squeeze_dims::<2>(&[-1]);
        let squeeze2_out1 = split6_out2.squeeze_dims::<2>(&[-1]);
        let squeeze3_out1 = split6_out3.squeeze_dims::<2>(&[-1]);
        let squeeze4_out1 = split6_out4.squeeze_dims::<2>(&[-1]);
        let add49_out1 = squeeze1_out1.clone().add(squeeze3_out1.clone());
        let div11_out1 = add49_out1
            .div((constant351_out1.clone()).unsqueeze_dims(&[0isize]));
        let add50_out1 = squeeze2_out1.clone().add(squeeze4_out1.clone());
        let div12_out1 = add50_out1
            .div((constant351_out1.clone()).unsqueeze_dims(&[0isize]));
        let sub11_out1 = squeeze3_out1.sub(squeeze1_out1);
        let sub12_out1 = squeeze4_out1.sub(squeeze2_out1);
        let unsqueeze17_out1: Tensor<3> = div11_out1.unsqueeze_dims::<3>(&[-1]);
        let unsqueeze18_out1: Tensor<3> = div12_out1.unsqueeze_dims::<3>(&[-1]);
        let unsqueeze19_out1: Tensor<3> = sub11_out1.unsqueeze_dims::<3>(&[-1]);
        let unsqueeze20_out1: Tensor<3> = sub12_out1.unsqueeze_dims::<3>(&[-1]);
        let concat26_out1 = burn::tensor::Tensor::cat(
            [unsqueeze17_out1, unsqueeze18_out1, unsqueeze19_out1, unsqueeze20_out1]
                .into(),
            2,
        );
        let linear29_out1 = self.linear29.forward(concat26_out1.clone());
        let relu38_out1 = burn::tensor::activation::relu(linear29_out1);
        let linear30_out1 = self.linear30.forward(relu38_out1);
        let clip6_out1 = {
            let __clip_min = -10f64;
            let __clip_max = 10f64;
            linear30_out1.clamp(__clip_min, __clip_max)
        };
        let add51_out1 = add41_out1.clone().add(clip6_out1.clone());
        let transpose24_out1 = add51_out1.permute([1, 0, 2]);
        let transpose25_out1 = add41_out1.clone().permute([1, 0, 2]);
        let linear31_out1 = self.linear31.forward(transpose24_out1.clone());
        let linear32_out1 = self.linear32.forward(transpose24_out1);
        let linear33_out1 = self.linear33.forward(transpose25_out1);
        let reshape27_out1 = linear31_out1.reshape([64, -1, 16]);
        let transpose26_out1 = reshape27_out1.permute([1, 0, 2]);
        let reshape28_out1 = linear32_out1.reshape([64, -1, 16]);
        let reshape29_out1 = linear33_out1.reshape([64, -1, 16]);
        let transpose27_out1 = reshape29_out1.permute([1, 0, 2]);
        let mul74_out1 = transpose26_out1
            .mul((constant350_out1).unsqueeze_dims(&[0isize, 1isize]));
        let transpose28_out1 = reshape28_out1.permute([1, 2, 0]);
        let matmul37_out1 = mul74_out1.matmul(transpose28_out1);
        let softmax5_out1 = burn::tensor::activation::softmax(matmul37_out1, 2);
        let matmul38_out1 = softmax5_out1.matmul(transpose27_out1);
        let transpose29_out1 = matmul38_out1.permute([1, 0, 2]);
        let reshape30_out1 = transpose29_out1.reshape([-1, 128]);
        let linear34_out1 = self.linear34.forward(reshape30_out1);
        let reshape31_out1 = linear34_out1.reshape([64, -1, 128]);
        let transpose30_out1 = reshape31_out1.permute([1, 0, 2]);
        let add52_out1 = add41_out1.clone().add(transpose30_out1);
        let reducemean13_out1 = { add52_out1.clone().mean_dim(2usize) };
        let sub13_out1 = add52_out1.sub(reducemean13_out1);
        let pow7_out1 = sub13_out1
            .clone()
            .powf((constant351_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean14_out1 = { pow7_out1.mean_dim(2usize) };
        let add53_out1 = reducemean14_out1
            .add((constant352_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt7_out1 = add53_out1.sqrt();
        let div13_out1 = sub13_out1.div(sqrt7_out1);
        let constant16_out1 = self.constant16.val();
        let mul75_out1 = div13_out1
            .mul((constant16_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant17_out1 = self.constant17.val();
        let add54_out1 = mul75_out1
            .add((constant17_out1).unsqueeze_dims(&[0isize, 1isize]));
        let add55_out1 = add54_out1.clone().add(clip6_out1);
        let linear35_out1 = self.linear35.forward(add55_out1.clone());
        let reshape32_out1 = linear35_out1.reshape([-1, 64, 8, 18, 2]);
        let linear36_out1 = self.linear36.forward(add55_out1);
        let reshape33_out1 = linear36_out1.reshape([-1, 64, 8, 18]);
        let softmax6_out1 = burn::tensor::activation::softmax(reshape33_out1, 3);
        let mul76_out1 = reshape32_out1
            .mul((constant299_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let unsqueeze21_out1: Tensor<5> = concat26_out1.unsqueeze_dims::<5>(&[2, 3]);
        let slice12_out1 = unsqueeze21_out1.clone().slice(s![.., .., .., .., 2..]);
        let mul77_out1 = mul76_out1.mul(slice12_out1);
        let mul78_out1 = mul77_out1
            .mul((constant355_out1).unsqueeze_dims(&[0isize, 1isize, 2isize, 3isize]));
        let slice13_out1 = unsqueeze21_out1.slice(s![.., .., .., .., 0..2]);
        let add56_out1 = slice13_out1.add(mul78_out1);
        let mul79_out1 = add56_out1
            .mul(
                (constant351_out1.clone())
                    .unsqueeze_dims(&[0isize, 1isize, 2isize, 3isize]),
            );
        let sub14_out1 = mul79_out1
            .sub((constant354_out1).unsqueeze_dims(&[0isize, 1isize, 2isize, 3isize]));
        let transpose31_out1 = sub14_out1.permute([0, 2, 1, 3, 4]);
        let reshape34_out1 = transpose31_out1.reshape([-1, 64, 18, 2]);
        let split_tensors = reshape34_out1.split_with_sizes([6, 6, 6].into(), 2);
        let [split7_out1, split7_out2, split7_out3] = split_tensors.try_into().unwrap();
        let gridsample4_out1 = slice5_out1
            .grid_sample_2d(
                split7_out1,
                burn::tensor::ops::GridSampleOptions::new(
                        burn::tensor::ops::InterpolateMode::Bilinear,
                    )
                    .with_padding_mode(burn::tensor::ops::GridSamplePaddingMode::Zeros)
                    .with_align_corners(false),
            );
        let gridsample5_out1 = slice6_out1
            .grid_sample_2d(
                split7_out2,
                burn::tensor::ops::GridSampleOptions::new(
                        burn::tensor::ops::InterpolateMode::Bilinear,
                    )
                    .with_padding_mode(burn::tensor::ops::GridSamplePaddingMode::Zeros)
                    .with_align_corners(false),
            );
        let gridsample6_out1 = slice7_out1
            .grid_sample_2d(
                split7_out3,
                burn::tensor::ops::GridSampleOptions::new(
                        burn::tensor::ops::InterpolateMode::Bilinear,
                    )
                    .with_padding_mode(burn::tensor::ops::GridSamplePaddingMode::Zeros)
                    .with_align_corners(false),
            );
        let transpose32_out1 = softmax6_out1.permute([0, 2, 1, 3]);
        let reshape35_out1 = transpose32_out1.reshape([-1, 1, 64, 18]);
        let concat27_out1 = burn::tensor::Tensor::cat(
            [gridsample4_out1, gridsample5_out1, gridsample6_out1].into(),
            3,
        );
        let mul80_out1 = concat27_out1.mul(reshape35_out1);
        let reducesum2_out1 = {
            mul80_out1.sum_dim(3usize).squeeze_dims::<3usize>(&[3])
        };
        let reshape36_out1 = reducesum2_out1.reshape(concat22_out1);
        let transpose33_out1 = reshape36_out1.permute([0, 2, 1]);
        let concat28_out1 = burn::tensor::Tensor::cat(
            [add54_out1.clone(), transpose33_out1.clone()].into(),
            2,
        );
        let linear37_out1 = self.linear37.forward(concat28_out1);
        let sigmoid52_out1 = burn::tensor::activation::sigmoid(linear37_out1);
        let slice14_out1 = sigmoid52_out1.clone().slice(s![.., .., 0..128]);
        let slice15_out1 = sigmoid52_out1.slice(s![.., .., 128..256]);
        let mul81_out1 = slice14_out1.mul(add54_out1);
        let mul82_out1 = slice15_out1.mul(transpose33_out1);
        let add57_out1 = mul81_out1.add(mul82_out1);
        let reducemean15_out1 = { add57_out1.clone().mean_dim(2usize) };
        let sub15_out1 = add57_out1.sub(reducemean15_out1);
        let pow8_out1 = sub15_out1
            .clone()
            .powf((constant351_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean16_out1 = { pow8_out1.mean_dim(2usize) };
        let add58_out1 = reducemean16_out1
            .add((constant352_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt8_out1 = add58_out1.sqrt();
        let div14_out1 = sub15_out1.div(sqrt8_out1);
        let constant21_out1 = self.constant21.val();
        let mul83_out1 = div14_out1
            .mul((constant21_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant22_out1 = self.constant22.val();
        let add59_out1 = mul83_out1
            .add((constant22_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear38_out1 = self.linear38.forward(add59_out1.clone());
        let relu39_out1 = burn::tensor::activation::relu(linear38_out1);
        let linear39_out1 = self.linear39.forward(relu39_out1);
        let add60_out1 = add59_out1.add(linear39_out1);
        let clip7_out1 = {
            let __clip_min = -65504f64;
            let __clip_max = 65504f64;
            add60_out1.clamp(__clip_min, __clip_max)
        };
        let reducemean17_out1 = { clip7_out1.clone().mean_dim(2usize) };
        let sub16_out1 = clip7_out1.sub(reducemean17_out1);
        let pow9_out1 = sub16_out1
            .clone()
            .powf((constant351_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean18_out1 = { pow9_out1.mean_dim(2usize) };
        let add61_out1 = reducemean18_out1
            .add((constant352_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt9_out1 = add61_out1.sqrt();
        let div15_out1 = sub16_out1.div(sqrt9_out1);
        let constant25_out1 = self.constant25.val();
        let mul84_out1 = div15_out1
            .mul((constant25_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant26_out1 = self.constant26.val();
        let add62_out1 = mul84_out1
            .add((constant26_out1).unsqueeze_dims(&[0isize, 1isize]));
        let add63_out1 = add62_out1.clone().add(add41_out1);
        let linear40_out1 = self.linear40.forward(add63_out1);
        let relu40_out1 = burn::tensor::activation::relu(linear40_out1);
        let linear41_out1 = self.linear41.forward(relu40_out1);
        let relu41_out1 = burn::tensor::activation::relu(linear41_out1);
        let linear42_out1 = self.linear42.forward(relu41_out1);
        let add64_out1 = linear42_out1.add(linear28_out1);
        (
            add64_out1,
            constant369_out1,
            constant391_out1,
            div9_out1,
            gather10_out1,
            div10_out1,
            gather13_out1,
            add62_out1,
        )
    }
}
#[derive(Module, Debug)]
pub struct Submodule7 {
    linear43: Linear,
    linear44: Linear,
    linear45: Linear,
    linear46: Linear,
    linear47: Linear,
    linear48: Linear,
    constant29: burn::module::Param<Tensor<1>>,
    constant30: burn::module::Param<Tensor<1>>,
    linear49: Linear,
    linear50: Linear,
    linear51: Linear,
    constant34: burn::module::Param<Tensor<1>>,
    constant35: burn::module::Param<Tensor<1>>,
    linear52: Linear,
    linear53: Linear,
    constant38: burn::module::Param<Tensor<1>>,
    constant39: burn::module::Param<Tensor<1>>,
    linear54: Linear,
    linear55: Linear,
    linear56: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule7 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let linear43 = LinearConfig::new(4, 256).with_bias(true).init(device);
        let linear44 = LinearConfig::new(256, 128).with_bias(true).init(device);
        let linear45 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear46 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear47 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear48 = LinearConfig::new(128, 128)
            .with_bias(true)
            .with_layout(LinearLayout::Col)
            .init(device);
        let constant29: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [128].into(),
        );
        let constant30: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [128].into(),
        );
        let linear49 = LinearConfig::new(128, 288).with_bias(true).init(device);
        let linear50 = LinearConfig::new(128, 144).with_bias(true).init(device);
        let linear51 = LinearConfig::new(256, 256).with_bias(true).init(device);
        let constant34: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [128].into(),
        );
        let constant35: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [128].into(),
        );
        let linear52 = LinearConfig::new(128, 512).with_bias(true).init(device);
        let linear53 = LinearConfig::new(512, 128).with_bias(true).init(device);
        let constant38: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [128].into(),
        );
        let constant39: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                1,
            >::zeros([128], (device, burn::tensor::DType::F32)),
            device.clone(),
            false,
            [128].into(),
        );
        let linear54 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear55 = LinearConfig::new(128, 128).with_bias(true).init(device);
        let linear56 = LinearConfig::new(128, 132).with_bias(true).init(device);
        Self {
            linear43,
            linear44,
            linear45,
            linear46,
            linear47,
            linear48,
            constant29,
            constant30,
            linear49,
            linear50,
            linear51,
            constant34,
            constant35,
            linear52,
            linear53,
            constant38,
            constant39,
            linear54,
            linear55,
            linear56,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add64_out1: Tensor<3>,
        constant369_out1: Tensor<1>,
        constant359_out1: [i64; 1],
        constant346_out1: [i64; 1],
        constant391_out1: Tensor<1>,
        div9_out1: Tensor<2>,
        gather10_out1: Tensor<2>,
        div10_out1: Tensor<2>,
        gather13_out1: Tensor<2>,
        constant351_out1: Tensor<1>,
        add62_out1: Tensor<3>,
        constant350_out1: Tensor<1>,
        constant352_out1: Tensor<1>,
        constant299_out1: Tensor<2>,
        constant355_out1: Tensor<1>,
        constant354_out1: Tensor<1>,
        slice5_out1: Tensor<4>,
        slice6_out1: Tensor<4>,
        slice7_out1: Tensor<4>,
        concat22_out1: [i64; 3],
    ) -> (Tensor<2>, Tensor<2>, Tensor<2>, Tensor<2>, Tensor<3>, Tensor<3>, [i64; 3]) {
        let shape11_out1: [i64; 3] = {
            let axes = &add64_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather18_out1 = shape11_out1[0] as i64;
        let reshape37_out1 = add64_out1.clone().reshape([-1, 33]);
        let softmax7_out1 = burn::tensor::activation::softmax(reshape37_out1, 1);
        let matmul47_out1 = softmax7_out1
            .matmul(constant369_out1.clone().unsqueeze_dims(&[-1isize]))
            .squeeze_dim::<1usize>(1usize);
        let unsqueeze22_out1 = [gather18_out1 as i64];
        let concat29_out1: [i64; 3usize] = [
            &unsqueeze22_out1[..],
            &constant359_out1[..],
            &constant346_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape38_out1 = matmul47_out1.reshape(concat29_out1);
        let gather19_out1 = {
            let sliced = reshape38_out1.clone().slice(s![.., .., 0]);
            sliced.squeeze_dim::<2usize>(2)
        };
        let add65_out1 = (constant391_out1.clone())
            .unsqueeze_dims(&[0isize])
            .add(gather19_out1);
        let mul85_out1 = add65_out1.mul(div9_out1.clone());
        let sub17_out1 = gather10_out1.clone().sub(mul85_out1);
        let gather20_out1 = {
            let sliced = reshape38_out1.clone().slice(s![.., .., 1]);
            sliced.squeeze_dim::<2usize>(2)
        };
        let add66_out1 = (constant391_out1.clone())
            .unsqueeze_dims(&[0isize])
            .add(gather20_out1);
        let mul86_out1 = add66_out1.mul(div10_out1.clone());
        let sub18_out1 = gather13_out1.clone().sub(mul86_out1);
        let gather21_out1 = {
            let sliced = reshape38_out1.clone().slice(s![.., .., 2]);
            sliced.squeeze_dim::<2usize>(2)
        };
        let add67_out1 = (constant391_out1.clone())
            .unsqueeze_dims(&[0isize])
            .add(gather21_out1);
        let mul87_out1 = add67_out1.mul(div9_out1.clone());
        let add68_out1 = gather10_out1.clone().add(mul87_out1);
        let gather22_out1 = {
            let sliced = reshape38_out1.slice(s![.., .., 3]);
            sliced.squeeze_dim::<2usize>(2)
        };
        let add69_out1 = (constant391_out1.clone())
            .unsqueeze_dims(&[0isize])
            .add(gather22_out1);
        let mul88_out1 = add69_out1.mul(div10_out1.clone());
        let add70_out1 = gather13_out1.clone().add(mul88_out1);
        let unsqueeze23_out1: Tensor<3> = sub17_out1.unsqueeze_dims::<3>(&[-1]);
        let unsqueeze24_out1: Tensor<3> = sub18_out1.unsqueeze_dims::<3>(&[-1]);
        let unsqueeze25_out1: Tensor<3> = add68_out1.unsqueeze_dims::<3>(&[-1]);
        let unsqueeze26_out1: Tensor<3> = add70_out1.unsqueeze_dims::<3>(&[-1]);
        let concat30_out1 = burn::tensor::Tensor::cat(
            [unsqueeze23_out1, unsqueeze24_out1, unsqueeze25_out1, unsqueeze26_out1]
                .into(),
            2,
        );
        let split_tensors = concat30_out1.split_with_sizes([1, 1, 1, 1].into(), 2);
        let [split8_out1, split8_out2, split8_out3, split8_out4] = split_tensors
            .try_into()
            .unwrap();
        let squeeze5_out1 = split8_out1.squeeze_dims::<2>(&[-1]);
        let squeeze6_out1 = split8_out2.squeeze_dims::<2>(&[-1]);
        let squeeze7_out1 = split8_out3.squeeze_dims::<2>(&[-1]);
        let squeeze8_out1 = split8_out4.squeeze_dims::<2>(&[-1]);
        let add71_out1 = squeeze5_out1.clone().add(squeeze7_out1.clone());
        let div16_out1 = add71_out1
            .div((constant351_out1.clone()).unsqueeze_dims(&[0isize]));
        let add72_out1 = squeeze6_out1.clone().add(squeeze8_out1.clone());
        let div17_out1 = add72_out1
            .div((constant351_out1.clone()).unsqueeze_dims(&[0isize]));
        let sub19_out1 = squeeze7_out1.sub(squeeze5_out1);
        let sub20_out1 = squeeze8_out1.sub(squeeze6_out1);
        let unsqueeze27_out1: Tensor<3> = div16_out1.unsqueeze_dims::<3>(&[-1]);
        let unsqueeze28_out1: Tensor<3> = div17_out1.unsqueeze_dims::<3>(&[-1]);
        let unsqueeze29_out1: Tensor<3> = sub19_out1.unsqueeze_dims::<3>(&[-1]);
        let unsqueeze30_out1: Tensor<3> = sub20_out1.unsqueeze_dims::<3>(&[-1]);
        let concat31_out1 = burn::tensor::Tensor::cat(
            [unsqueeze27_out1, unsqueeze28_out1, unsqueeze29_out1, unsqueeze30_out1]
                .into(),
            2,
        );
        let linear43_out1 = self.linear43.forward(concat31_out1.clone());
        let relu42_out1 = burn::tensor::activation::relu(linear43_out1);
        let linear44_out1 = self.linear44.forward(relu42_out1);
        let clip8_out1 = {
            let __clip_min = -10f64;
            let __clip_max = 10f64;
            linear44_out1.clamp(__clip_min, __clip_max)
        };
        let add73_out1 = add62_out1.clone().add(clip8_out1.clone());
        let transpose34_out1 = add73_out1.permute([1, 0, 2]);
        let transpose35_out1 = add62_out1.clone().permute([1, 0, 2]);
        let linear45_out1 = self.linear45.forward(transpose34_out1.clone());
        let linear46_out1 = self.linear46.forward(transpose34_out1);
        let linear47_out1 = self.linear47.forward(transpose35_out1);
        let reshape39_out1 = linear45_out1.reshape([64, -1, 16]);
        let transpose36_out1 = reshape39_out1.permute([1, 0, 2]);
        let reshape40_out1 = linear46_out1.reshape([64, -1, 16]);
        let reshape41_out1 = linear47_out1.reshape([64, -1, 16]);
        let transpose37_out1 = reshape41_out1.permute([1, 0, 2]);
        let mul89_out1 = transpose36_out1
            .mul((constant350_out1).unsqueeze_dims(&[0isize, 1isize]));
        let transpose38_out1 = reshape40_out1.permute([1, 2, 0]);
        let matmul53_out1 = mul89_out1.matmul(transpose38_out1);
        let softmax8_out1 = burn::tensor::activation::softmax(matmul53_out1, 2);
        let matmul54_out1 = softmax8_out1.matmul(transpose37_out1);
        let transpose39_out1 = matmul54_out1.permute([1, 0, 2]);
        let reshape42_out1 = transpose39_out1.reshape([-1, 128]);
        let linear48_out1 = self.linear48.forward(reshape42_out1);
        let reshape43_out1 = linear48_out1.reshape([64, -1, 128]);
        let transpose40_out1 = reshape43_out1.permute([1, 0, 2]);
        let add74_out1 = add62_out1.clone().add(transpose40_out1);
        let reducemean19_out1 = { add74_out1.clone().mean_dim(2usize) };
        let sub21_out1 = add74_out1.sub(reducemean19_out1);
        let pow10_out1 = sub21_out1
            .clone()
            .powf((constant351_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean20_out1 = { pow10_out1.mean_dim(2usize) };
        let add75_out1 = reducemean20_out1
            .add((constant352_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt10_out1 = add75_out1.sqrt();
        let div18_out1 = sub21_out1.div(sqrt10_out1);
        let constant29_out1 = self.constant29.val();
        let mul90_out1 = div18_out1
            .mul((constant29_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant30_out1 = self.constant30.val();
        let add76_out1 = mul90_out1
            .add((constant30_out1).unsqueeze_dims(&[0isize, 1isize]));
        let add77_out1 = add76_out1.clone().add(clip8_out1);
        let linear49_out1 = self.linear49.forward(add77_out1.clone());
        let reshape44_out1 = linear49_out1.reshape([-1, 64, 8, 18, 2]);
        let linear50_out1 = self.linear50.forward(add77_out1);
        let reshape45_out1 = linear50_out1.reshape([-1, 64, 8, 18]);
        let softmax9_out1 = burn::tensor::activation::softmax(reshape45_out1, 3);
        let mul91_out1 = reshape44_out1
            .mul((constant299_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let unsqueeze31_out1: Tensor<5> = concat31_out1.unsqueeze_dims::<5>(&[2, 3]);
        let slice16_out1 = unsqueeze31_out1.clone().slice(s![.., .., .., .., 2..]);
        let mul92_out1 = mul91_out1.mul(slice16_out1);
        let mul93_out1 = mul92_out1
            .mul((constant355_out1).unsqueeze_dims(&[0isize, 1isize, 2isize, 3isize]));
        let slice17_out1 = unsqueeze31_out1.slice(s![.., .., .., .., 0..2]);
        let add78_out1 = slice17_out1.add(mul93_out1);
        let mul94_out1 = add78_out1
            .mul(
                (constant351_out1.clone())
                    .unsqueeze_dims(&[0isize, 1isize, 2isize, 3isize]),
            );
        let sub22_out1 = mul94_out1
            .sub((constant354_out1).unsqueeze_dims(&[0isize, 1isize, 2isize, 3isize]));
        let transpose41_out1 = sub22_out1.permute([0, 2, 1, 3, 4]);
        let reshape46_out1 = transpose41_out1.reshape([-1, 64, 18, 2]);
        let split_tensors = reshape46_out1.split_with_sizes([6, 6, 6].into(), 2);
        let [split9_out1, split9_out2, split9_out3] = split_tensors.try_into().unwrap();
        let gridsample7_out1 = slice5_out1
            .grid_sample_2d(
                split9_out1,
                burn::tensor::ops::GridSampleOptions::new(
                        burn::tensor::ops::InterpolateMode::Bilinear,
                    )
                    .with_padding_mode(burn::tensor::ops::GridSamplePaddingMode::Zeros)
                    .with_align_corners(false),
            );
        let gridsample8_out1 = slice6_out1
            .grid_sample_2d(
                split9_out2,
                burn::tensor::ops::GridSampleOptions::new(
                        burn::tensor::ops::InterpolateMode::Bilinear,
                    )
                    .with_padding_mode(burn::tensor::ops::GridSamplePaddingMode::Zeros)
                    .with_align_corners(false),
            );
        let gridsample9_out1 = slice7_out1
            .grid_sample_2d(
                split9_out3,
                burn::tensor::ops::GridSampleOptions::new(
                        burn::tensor::ops::InterpolateMode::Bilinear,
                    )
                    .with_padding_mode(burn::tensor::ops::GridSamplePaddingMode::Zeros)
                    .with_align_corners(false),
            );
        let transpose42_out1 = softmax9_out1.permute([0, 2, 1, 3]);
        let reshape47_out1 = transpose42_out1.reshape([-1, 1, 64, 18]);
        let concat32_out1 = burn::tensor::Tensor::cat(
            [gridsample7_out1, gridsample8_out1, gridsample9_out1].into(),
            3,
        );
        let mul95_out1 = concat32_out1.mul(reshape47_out1);
        let reducesum3_out1 = {
            mul95_out1.sum_dim(3usize).squeeze_dims::<3usize>(&[3])
        };
        let reshape48_out1 = reducesum3_out1.reshape(concat22_out1);
        let transpose43_out1 = reshape48_out1.permute([0, 2, 1]);
        let concat33_out1 = burn::tensor::Tensor::cat(
            [add76_out1.clone(), transpose43_out1.clone()].into(),
            2,
        );
        let linear51_out1 = self.linear51.forward(concat33_out1);
        let sigmoid53_out1 = burn::tensor::activation::sigmoid(linear51_out1);
        let slice18_out1 = sigmoid53_out1.clone().slice(s![.., .., 0..128]);
        let slice19_out1 = sigmoid53_out1.slice(s![.., .., 128..256]);
        let mul96_out1 = slice18_out1.mul(add76_out1);
        let mul97_out1 = slice19_out1.mul(transpose43_out1);
        let add79_out1 = mul96_out1.add(mul97_out1);
        let reducemean21_out1 = { add79_out1.clone().mean_dim(2usize) };
        let sub23_out1 = add79_out1.sub(reducemean21_out1);
        let pow11_out1 = sub23_out1
            .clone()
            .powf((constant351_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean22_out1 = { pow11_out1.mean_dim(2usize) };
        let add80_out1 = reducemean22_out1
            .add((constant352_out1.clone()).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt11_out1 = add80_out1.sqrt();
        let div19_out1 = sub23_out1.div(sqrt11_out1);
        let constant34_out1 = self.constant34.val();
        let mul98_out1 = div19_out1
            .mul((constant34_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant35_out1 = self.constant35.val();
        let add81_out1 = mul98_out1
            .add((constant35_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear52_out1 = self.linear52.forward(add81_out1.clone());
        let relu43_out1 = burn::tensor::activation::relu(linear52_out1);
        let linear53_out1 = self.linear53.forward(relu43_out1);
        let add82_out1 = add81_out1.add(linear53_out1);
        let clip9_out1 = {
            let __clip_min = -65504f64;
            let __clip_max = 65504f64;
            add82_out1.clamp(__clip_min, __clip_max)
        };
        let reducemean23_out1 = { clip9_out1.clone().mean_dim(2usize) };
        let sub24_out1 = clip9_out1.sub(reducemean23_out1);
        let pow12_out1 = sub24_out1
            .clone()
            .powf((constant351_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean24_out1 = { pow12_out1.mean_dim(2usize) };
        let add83_out1 = reducemean24_out1
            .add((constant352_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt12_out1 = add83_out1.sqrt();
        let div20_out1 = sub24_out1.div(sqrt12_out1);
        let constant38_out1 = self.constant38.val();
        let mul99_out1 = div20_out1
            .mul((constant38_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant39_out1 = self.constant39.val();
        let add84_out1 = mul99_out1
            .add((constant39_out1).unsqueeze_dims(&[0isize, 1isize]));
        let add85_out1 = add84_out1.clone().add(add62_out1);
        let linear54_out1 = self.linear54.forward(add85_out1);
        let relu44_out1 = burn::tensor::activation::relu(linear54_out1);
        let linear55_out1 = self.linear55.forward(relu44_out1);
        let relu45_out1 = burn::tensor::activation::relu(linear55_out1);
        let linear56_out1 = self.linear56.forward(relu45_out1);
        let add86_out1 = linear56_out1.add(add64_out1);
        let shape12_out1: [i64; 3] = {
            let axes = &add86_out1.clone().dims()[0..3];
            let mut output = [0i64; 3];
            for i in 0..3 {
                output[i] = axes[i] as i64;
            }
            output
        };
        let gather23_out1 = shape12_out1[0] as i64;
        let reshape49_out1 = add86_out1.clone().reshape([-1, 33]);
        let softmax10_out1 = burn::tensor::activation::softmax(reshape49_out1, 1);
        let matmul63_out1 = softmax10_out1
            .matmul(constant369_out1.unsqueeze_dims(&[-1isize]))
            .squeeze_dim::<1usize>(1usize);
        let unsqueeze32_out1 = [gather23_out1 as i64];
        let concat34_out1: [i64; 3usize] = [
            &unsqueeze32_out1[..],
            &constant359_out1[..],
            &constant346_out1[..],
        ]
            .concat()
            .try_into()
            .unwrap();
        let reshape50_out1 = matmul63_out1.reshape(concat34_out1);
        let gather24_out1 = {
            let sliced = reshape50_out1.clone().slice(s![.., .., 0]);
            sliced.squeeze_dim::<2usize>(2)
        };
        let add87_out1 = (constant391_out1.clone())
            .unsqueeze_dims(&[0isize])
            .add(gather24_out1);
        let mul100_out1 = add87_out1.mul(div9_out1.clone());
        let sub25_out1 = gather10_out1.clone().sub(mul100_out1);
        let gather25_out1 = {
            let sliced = reshape50_out1.clone().slice(s![.., .., 1]);
            sliced.squeeze_dim::<2usize>(2)
        };
        let add88_out1 = (constant391_out1.clone())
            .unsqueeze_dims(&[0isize])
            .add(gather25_out1);
        let mul101_out1 = add88_out1.mul(div10_out1.clone());
        let sub26_out1 = gather13_out1.sub(mul101_out1);
        let gather26_out1 = {
            let sliced = reshape50_out1.clone().slice(s![.., .., 2]);
            sliced.squeeze_dim::<2usize>(2)
        };
        let add89_out1 = (constant391_out1.clone())
            .unsqueeze_dims(&[0isize])
            .add(gather26_out1);
        let mul102_out1 = add89_out1.mul(div9_out1);
        let add90_out1 = gather10_out1.add(mul102_out1);
        let gather27_out1 = {
            let sliced = reshape50_out1.slice(s![.., .., 3]);
            sliced.squeeze_dim::<2usize>(2)
        };
        let add91_out1 = (constant391_out1).unsqueeze_dims(&[0isize]).add(gather27_out1);
        let mul103_out1 = add91_out1.mul(div10_out1);
        (
            mul103_out1,
            sub25_out1,
            sub26_out1,
            add90_out1,
            add84_out1,
            add86_out1,
            concat34_out1,
        )
    }
}
#[derive(Module, Debug)]
pub struct Submodule8 {
    linear57: Linear,
    linear58: Linear,
    linear59: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule8 {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let linear57 = LinearConfig::new(128, 1).with_bias(true).init(device);
        let linear58 = LinearConfig::new(20, 64).with_bias(true).init(device);
        let linear59 = LinearConfig::new(64, 1).with_bias(true).init(device);
        Self {
            linear57,
            linear58,
            linear59,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        gather13_out1: Tensor<2>,
        mul103_out1: Tensor<2>,
        sub25_out1: Tensor<2>,
        sub26_out1: Tensor<2>,
        add90_out1: Tensor<2>,
        constant351_out1: Tensor<1>,
        add84_out1: Tensor<3>,
        add86_out1: Tensor<3>,
        concat34_out1: [i64; 3],
        constant355_out1: Tensor<1>,
        orig_target_sizes: Tensor<2, Int>,
    ) -> (Tensor<2, Int>, Tensor<3>, Tensor<2>) {
        let add92_out1 = gather13_out1.add(mul103_out1);
        let unsqueeze33_out1: Tensor<3> = sub25_out1.unsqueeze_dims::<3>(&[-1]);
        let unsqueeze34_out1: Tensor<3> = sub26_out1.unsqueeze_dims::<3>(&[-1]);
        let unsqueeze35_out1: Tensor<3> = add90_out1.unsqueeze_dims::<3>(&[-1]);
        let unsqueeze36_out1: Tensor<3> = add92_out1.unsqueeze_dims::<3>(&[-1]);
        let concat35_out1 = burn::tensor::Tensor::cat(
            [unsqueeze33_out1, unsqueeze34_out1, unsqueeze35_out1, unsqueeze36_out1]
                .into(),
            2,
        );
        let split_tensors = concat35_out1.split_with_sizes([1, 1, 1, 1].into(), 2);
        let [split10_out1, split10_out2, split10_out3, split10_out4] = split_tensors
            .try_into()
            .unwrap();
        let squeeze9_out1 = split10_out1.squeeze_dims::<2>(&[-1]);
        let squeeze10_out1 = split10_out2.squeeze_dims::<2>(&[-1]);
        let squeeze11_out1 = split10_out3.squeeze_dims::<2>(&[-1]);
        let squeeze12_out1 = split10_out4.squeeze_dims::<2>(&[-1]);
        let add93_out1 = squeeze9_out1.clone().add(squeeze11_out1.clone());
        let div21_out1 = add93_out1
            .div((constant351_out1.clone()).unsqueeze_dims(&[0isize]));
        let add94_out1 = squeeze10_out1.clone().add(squeeze12_out1.clone());
        let div22_out1 = add94_out1.div((constant351_out1).unsqueeze_dims(&[0isize]));
        let sub27_out1 = squeeze11_out1.sub(squeeze9_out1);
        let sub28_out1 = squeeze12_out1.sub(squeeze10_out1);
        let unsqueeze37_out1: Tensor<3> = div21_out1.unsqueeze_dims::<3>(&[-1]);
        let unsqueeze38_out1: Tensor<3> = div22_out1.unsqueeze_dims::<3>(&[-1]);
        let unsqueeze39_out1: Tensor<3> = sub27_out1.unsqueeze_dims::<3>(&[-1]);
        let unsqueeze40_out1: Tensor<3> = sub28_out1.unsqueeze_dims::<3>(&[-1]);
        let concat36_out1 = burn::tensor::Tensor::cat(
            [unsqueeze37_out1, unsqueeze38_out1, unsqueeze39_out1, unsqueeze40_out1]
                .into(),
            2,
        );
        let linear57_out1 = self.linear57.forward(add84_out1);
        let reshape51_out1 = add86_out1.reshape([-1, 64, 4, 33]);
        let softmax11_out1 = burn::tensor::activation::softmax(reshape51_out1, 3);
        let (topk2_out1, __topk_indices_raw) = softmax11_out1.topk_with_indices(4, 3);
        let topk2_out2 = __topk_indices_raw.cast(burn::tensor::DType::I64);
        let reducemean25_out1 = { topk2_out1.clone().mean_dim(3usize) };
        let concat37_out1 = burn::tensor::Tensor::cat(
            [topk2_out1, reducemean25_out1].into(),
            3,
        );
        let reshape52_out1 = concat37_out1.reshape(concat34_out1);
        let linear58_out1 = self.linear58.forward(reshape52_out1);
        let relu46_out1 = burn::tensor::activation::relu(linear58_out1);
        let linear59_out1 = self.linear59.forward(relu46_out1);
        let add95_out1 = linear57_out1.add(linear59_out1);
        let unsqueeze41_out1: Tensor<4> = concat36_out1.unsqueeze_dims::<4>(&[0]);
        let unsqueeze42_out1: Tensor<4> = add95_out1.unsqueeze_dims::<4>(&[0]);
        let gather28_out1 = {
            let sliced = unsqueeze42_out1.slice(s![- 1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let gather29_out1 = {
            let sliced = unsqueeze41_out1.slice(s![- 1, .., .., ..]);
            sliced.squeeze_dim::<3usize>(0)
        };
        let split_tensors = gather29_out1.split_with_sizes([1, 1, 1, 1].into(), 2);
        let [split11_out1, split11_out2, split11_out3, split11_out4] = split_tensors
            .try_into()
            .unwrap();
        let squeeze13_out1 = split11_out1.squeeze_dims::<2>(&[-1]);
        let squeeze14_out1 = split11_out2.squeeze_dims::<2>(&[-1]);
        let squeeze15_out1 = split11_out3.squeeze_dims::<2>(&[-1]);
        let squeeze16_out1 = split11_out4.squeeze_dims::<2>(&[-1]);
        let mul104_out1 = squeeze15_out1
            .mul((constant355_out1.clone()).unsqueeze_dims(&[0isize]));
        let sub29_out1 = squeeze13_out1.clone().sub(mul104_out1.clone());
        let mul105_out1 = squeeze16_out1
            .mul((constant355_out1).unsqueeze_dims(&[0isize]));
        let sub30_out1 = squeeze14_out1.clone().sub(mul105_out1.clone());
        let add96_out1 = squeeze13_out1.add(mul104_out1);
        let add97_out1 = squeeze14_out1.add(mul105_out1);
        let unsqueeze43_out1: Tensor<3> = sub29_out1.unsqueeze_dims::<3>(&[-1]);
        let unsqueeze44_out1: Tensor<3> = sub30_out1.unsqueeze_dims::<3>(&[-1]);
        let unsqueeze45_out1: Tensor<3> = add96_out1.unsqueeze_dims::<3>(&[-1]);
        let unsqueeze46_out1: Tensor<3> = add97_out1.unsqueeze_dims::<3>(&[-1]);
        let concat38_out1 = burn::tensor::Tensor::cat(
            [unsqueeze43_out1, unsqueeze44_out1, unsqueeze45_out1, unsqueeze46_out1]
                .into(),
            2,
        );
        let tile4_out1 = orig_target_sizes.repeat(&[1, 2]);
        let unsqueeze47_out1: Tensor<3, Int> = tile4_out1.unsqueeze_dims::<3>(&[1]);
        let cast1_out1 = unsqueeze47_out1.float().cast(burn::tensor::DType::F32);
        let mul106_out1 = concat38_out1.mul(cast1_out1);
        let sigmoid54_out1 = burn::tensor::activation::sigmoid(gather28_out1);
        let flatten1_out1 = {
            let leading_dim = sigmoid54_out1.shape()[..1].iter().product::<usize>()
                as i32;
            sigmoid54_out1.reshape::<2, _>([leading_dim, -1])
        };
        let (topk3_out1, __topk_indices_raw) = flatten1_out1.topk_with_indices(64, 1);
        let topk3_out2 = __topk_indices_raw.cast(burn::tensor::DType::I64);
        let sub31_out1 = topk3_out2.clone().sub(topk3_out2.clone());
        let unsqueeze48_out1: Tensor<3, Int> = topk3_out2.unsqueeze_dims::<3>(&[-1]);
        let tile5_out1 = unsqueeze48_out1.repeat(&[1, 1, 4]);
        let gatherelements3_out1 = mul106_out1.gather(1, tile5_out1);
        (sub31_out1, gatherelements3_out1, topk3_out1)
    }
}

#[derive(Module, Debug)]
pub struct Model {
    submodule1: Submodule1,
    submodule2: Submodule2,
    submodule3: Submodule3,
    submodule4: Submodule4,
    submodule5: Submodule5,
    submodule6: Submodule6,
    submodule7: Submodule7,
    submodule8: Submodule8,
    #[module(skip)]
    device: Device,
}


extern crate std;

impl Default for Model {
    fn default() -> Self {
        Self::from_file(
            "burn_models/detect_fresh/meiki.text.detect.v0.1.bpk",
            &Default::default(),
        )
    }
}

impl Model {
    /// Load model weights from a burnpack file.
    pub fn from_file<P: AsRef<std::path::Path>>(file: P, device: &Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_file(file);
        model.load_from(&mut store).expect("Failed to load burnpack file");
        model
    }

    /// Load model weights from in-memory bytes.
    ///
    /// The bytes must be the contents of a `.bpk` file.
    pub fn from_bytes(bytes: Bytes, device: &Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_bytes(Some(bytes));
        model.load_from(&mut store).expect("Failed to load burnpack bytes");
        model
    }
}

impl Model {
    #[allow(unused_variables)]
    pub fn new(device: &Device) -> Self {
        let submodule1 = Submodule1::new(device);
        let submodule2 = Submodule2::new(device);
        let submodule3 = Submodule3::new(device);
        let submodule4 = Submodule4::new(device);
        let submodule5 = Submodule5::new(device);
        let submodule6 = Submodule6::new(device);
        let submodule7 = Submodule7::new(device);
        let submodule8 = Submodule8::new(device);
        Self {
            submodule1,
            submodule2,
            submodule3,
            submodule4,
            submodule5,
            submodule6,
            submodule7,
            submodule8,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        images: Tensor<4>,
        orig_target_sizes: Tensor<2, Int>,
    ) -> (Tensor<2, Int>, Tensor<3>, Tensor<2>) {
        let (add7_out1, relu5_out1, add5_out1) = self.submodule1.forward(images);
        let (
            sub1_out1,
            conv2d47_out1,
            conv2d46_out1,
            constant346_out1,
            constant350_out1,
        ) = self.submodule2.forward(add7_out1, relu5_out1, add5_out1);
        let (
            concat14_out1,
            constant351_out1,
            constant352_out1,
            constant355_out1,
            constant354_out1,
        ) = self
            .submodule3
            .forward(sub1_out1, conv2d47_out1, conv2d46_out1, constant346_out1.clone());
        let (
            add31_out1,
            clip1_out1,
            sigmoid49_out1,
            slice5_out1,
            slice6_out1,
            slice7_out1,
        ) = self
            .submodule4
            .forward(
                concat14_out1,
                constant351_out1.clone(),
                constant352_out1.clone(),
                constant346_out1.clone(),
                constant350_out1.clone(),
            );
        let (div6_out1, constant359_out1, constant299_out1, concat22_out1) = self
            .submodule5
            .forward(
                add31_out1,
                constant351_out1.clone(),
                constant352_out1.clone(),
                clip1_out1,
                sigmoid49_out1.clone(),
                constant355_out1.clone(),
                slice5_out1.clone(),
                constant354_out1.clone(),
                slice6_out1.clone(),
                slice7_out1.clone(),
            );
        let (
            add64_out1,
            constant369_out1,
            constant391_out1,
            div9_out1,
            gather10_out1,
            div10_out1,
            gather13_out1,
            add62_out1,
        ) = self
            .submodule6
            .forward(
                div6_out1,
                constant351_out1.clone(),
                constant352_out1.clone(),
                sigmoid49_out1,
                constant354_out1.clone(),
                constant359_out1.clone(),
                constant346_out1.clone(),
                constant350_out1.clone(),
                constant299_out1.clone(),
                constant355_out1.clone(),
                slice5_out1.clone(),
                slice6_out1.clone(),
                slice7_out1.clone(),
                concat22_out1.clone(),
            );
        let (
            mul103_out1,
            sub25_out1,
            sub26_out1,
            add90_out1,
            add84_out1,
            add86_out1,
            concat34_out1,
        ) = self
            .submodule7
            .forward(
                add64_out1,
                constant369_out1,
                constant359_out1,
                constant346_out1,
                constant391_out1,
                div9_out1,
                gather10_out1,
                div10_out1,
                gather13_out1.clone(),
                constant351_out1.clone(),
                add62_out1,
                constant350_out1,
                constant352_out1,
                constant299_out1,
                constant355_out1.clone(),
                constant354_out1,
                slice5_out1,
                slice6_out1,
                slice7_out1,
                concat22_out1,
            );
        let (sub31_out1, gatherelements3_out1, topk3_out1) = self
            .submodule8
            .forward(
                gather13_out1,
                mul103_out1,
                sub25_out1,
                sub26_out1,
                add90_out1,
                constant351_out1,
                add84_out1,
                add86_out1,
                concat34_out1,
                constant355_out1,
                orig_target_sizes,
            );
        (sub31_out1, gatherelements3_out1, topk3_out1)
    }
}
