from src.model import ViT
import math
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import src.config
import time


def float2fix(value, frac_len=0, word_len=8, round_method='floor'):
    min_value = -2 ** (word_len - 1)
    max_value = 2 ** (word_len - 1) - 1

    if round_method == 'round':
        fix_value = np.floor(value * (2 ** frac_len) + 0.5)
    else:
        fix_value = np.floor(value * (2 ** frac_len))

    fix_value[fix_value < min_value] = min_value
    fix_value[fix_value > max_value] = max_value
    fix_value = fix_value / (2 ** frac_len)
    return fix_value



class validation:
    def load_model(self):
        # Load ViT
        print('The version of pytorch', torch.__version__)
        print('Accessibility of CUDA: ', torch.cuda.is_available())
        model = ViT('B_16_imagenet1k', pretrained=True)
        model.cuda()
        model.eval()
        # Load image
        testdir = 'partial_imagenet/imagenet_images'
        # testdir = 'img_dir'
        test_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(testdir, test_transforms),
            batch_size=16, # 16
            shuffle=False,
            num_workers=0, pin_memory=True
        )
        print("Total number of batches:", len(test_loader))
        self.model = model
        self.test_loader = test_loader
        return

    def test_unquantized(self, batch_num=10):
        src.config.do_quantization = False
        src.config.use_static_value = False
        model, test_loader = self.model, self.test_loader
        print("Before quantization:")
        with torch.no_grad():
            correct_num = 0
            total_image_num = 0
            for i, (images, target) in enumerate(test_loader):
                if i == batch_num:
                    break
                images = images.to('cuda')
                output = model(images)
                output = output.cpu().detach()
                pred = torch.argmax(output, axis=1)
                # print(torch.max(output, axis=1))
                correct_num += torch.sum(pred == target).numpy()
                total_image_num += len(target)
                test_acc = correct_num / total_image_num
                print("batch #%d, test_acc:%f" % (i + 1, test_acc))

    def model_weight_quantization(self):
        model = self.model
        state_dict = model.state_dict()
        for key in state_dict.keys():
            numpy_val = state_dict[key].cpu().numpy()
            max_v = numpy_val.max()
            min_v = numpy_val.min()
            max_v = math.ceil(max_v)
            min_v = math.floor(min_v)
            int_part = max(abs(max_v), abs(min_v))
            int_bits = math.ceil(np.log2(int_part)) if int_part != 0 else 0
            mantisaa_bits = 16 - 1 - int_bits
            new_np_array = float2fix(numpy_val, frac_len=mantisaa_bits, word_len=16, round_method='floor')
            state_dict[key] = torch.from_numpy(new_np_array)
            # print(key, 'acc_loss=', np.mean(numpy_val - new_np_array))
        self.model.load_state_dict(state_dict)
        self.model.cuda()
        self.model.eval()


    def search_optimal_quantization_params(self, batch_num=5):
        quantize_param_dict = src.config.quantize_param_dict
        model, test_loader = self.model, self.test_loader
        optimal_setting = dict()
        with torch.no_grad():
            for i, (images, target) in enumerate(test_loader):
                if i == batch_num:
                    break
                images = images.to('cuda')
                model(images)
                for key in quantize_param_dict:
                    if key in optimal_setting:
                        optimal_setting[key] = min(optimal_setting[key], quantize_param_dict[key])
                    else:
                        optimal_setting[key] = quantize_param_dict[key]
        return optimal_setting

    def test_quantized(self, batch_num=1):
        model, test_loader = self.model, self.test_loader
        print("After quantization:")
        src.config.do_quantization = True
        src.config.use_static_value = True
        with torch.no_grad():
            correct_num = 0
            total_image_num = 0
            for i, (images, target) in enumerate(test_loader):
                # if i == batch_num:
                #     break
                images = images.to('cuda')
                output = model(images)
                output = output.cpu().detach()
                pred = torch.argmax(output, axis=1)
                # print(torch.max(output, axis=1))
                correct_num += torch.sum(pred == target).numpy()
                total_image_num += len(target)
                test_acc = correct_num / total_image_num
                print("batch #%d, test_acc:%f" % (i + 1, test_acc))

    def update_para_setting(self, setting):
        src.config.quantize_param_dict = setting

if __name__ == '__main__':
    obj = validation()
    obj.load_model()

    # obj.test_unquantized(batch_num=1)

    obj.model_weight_quantization()

    # optimal_setting = obj.search_optimal_quantization_params(batch_num=10)
    # print('The searched optimal setting is:', optimal_setting)

    # obj.update_para_setting(optimal_setting)

    src.config.quan_in = 6
    src.config.quan_out = 4
    s = time.time()
    obj.test_quantized(batch_num=1)
    e = time.time()
    print('inference time: ', e - s)
    exit()




# The following parts are the quantization parameter search for gelu (pwlinear version)
    # f = open('tmp.txt', 'w+')
    # for a in range(7, 12):
    #     for b in range(1, 12):
    #         print('cur_config is ', a ,'  ',b)
    #         src.config.quan_in = a
    #         src.config.quan_out = b
    #         src.config.accumulative_err = 0
    #
    #         s = time.time()
    #         obj.test_quantized(batch_num=1)
    #         e = time.time()
    #         print('inference time: ', e - s)
    #         print('accumulative_err under this setting is: ', src.config.accumulative_err)
    #         f.write('%d %d %f\n'%(a, b,  src.config.accumulative_err))
    # f.close()