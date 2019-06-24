import torch
from torch.autograd import Variable
from lib.img_dataset import LandmarkDataset


class ImgEmbedder:

    def __init__(self, model, dir_images):
        self.model = model
        self.dataset = LandmarkDataset(dir_images)

    def get_vector(self, file_name, pos=0, size=288):

        input_tensor = self.dataset.get_item_pos(file_name, pos, size=size)
        input_tensor = torch.unsqueeze(input_tensor, dim=0).cuda()
        output = self.model.forward(Variable(input_tensor))
        output_data = output.data.cpu().numpy()[0]

        return output_data
