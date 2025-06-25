import os

class Config:
    def __init__(self, model='resnet18', w_bit=4, a_bit=8, tbs=128, defaultMap=True):
        self.train_batch_size = tbs
        self.eval_batch_size = 100
        self.num_calibration_batches = 32

        self.model = model

        saved_path = 'quantized_models/'
        sub_path = f'{self.model}'
        model_path = os.path.join(saved_path, sub_path)
        os.makedirs(model_path, exist_ok=True)

        self.model_path = model_path
        # self.scripted_float_model_path = os.path.join(model_path, 'float_scripted.pth')
        # self.scripted_quantized_model_path = os.path.join(model_path, 'quantized_scripted.pth')
        self.qat_float_path = os.path.join(model_path, 'qat_float.pth')
        self.qat_quantized_path = os.path.join(model_path, 'qat_quantized.pth')

        self.weight_list_path = os.path.join(model_path, 'weight_list.pth')
        self.act_list_path = os.path.join(model_path, 'act_list.pth')

        self.w_bit = w_bit
        self.a_bit = a_bit
        self.qat_train_bs = 999999
        self.defaultMap = defaultMap  # 不使用默认的量化推理流程, 对量化有效
