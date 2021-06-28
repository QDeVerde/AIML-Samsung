import torch
import torchvision

from verde.utils import ensure_is_list


class InceptionExtractor:

    def __init__(self, /, device=torch.device('cpu')):
        self._model = torchvision.models.inception_v3(pretrained=True, aux_logits=True).to(device)
        self._device = device

        for params in self._model.parameters():
            params.requires_grad = False

        self._storage = None
        self._model.Mixed_7c.register_forward_hook(self._hook)

        self._model.eval()

        self._inception_preprocessing = torchvision.transforms.Compose([
            torchvision.transforms.Resize(299),
            torchvision.transforms.CenterCrop(299),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _hook(self, module, input, output):
        self._storage = output

    def _preprocess(self, image):
        return self._inception_preprocessing(image).to(self._device)

    def __call__(self, target):
        images = ensure_is_list(target)

        images = map(self._preprocess, images)

        images = map(lambda tensor: tensor.unsqueeze(0), images)

        images = list(images)

        stack = torch.cat(images, dim=0)

        with torch.no_grad():
            _ = self._model(stack)

        return self._storage
