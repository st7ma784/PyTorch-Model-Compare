import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from warnings import warn
from typing import List, Dict
import matplotlib.pyplot as plt

from mpl_toolkits import axes_grid1
import matplotlib.pyplot as plt


def add_colorbar(im, aspect=10, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)
class CKA:
    def __init__(self,
                 model1: nn.Module,
                 model2: nn.Module,
                 model1_name: str = None,
                 model2_name: str = None,
                 model1_layers: List[str] = None,
                 model2_layers: List[str] = None,
                 device: str ='cpu'):
        """

        :param model1: (nn.Module) Neural Network 1
        :param model2: (nn.Module) Neural Network 2
        :param model1_name: (str) Name of model 1
        :param model2_name: (str) Name of model 2
        :param model1_layers: (List) List of layers to extract features from
        :param model2_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model1 = model1
        self.model2 = model2

        self.device = device

        self.model1_info = {}
        self.model2_info = {}

        if model1_name is None:
            self.model1_info['Name'] = model1.__repr__().split('(')[0]
        else:
            self.model1_info['Name'] = model1_name

        if model2_name is None:
            self.model2_info['Name'] = model2.__repr__().split('(')[0]
        else:
            self.model2_info['Name'] = model2_name

        if self.model1_info['Name'] == self.model2_info['Name']:
            warn(f"Both model have identical names - {self.model2_info['Name']}. " \
                 "It may cause confusion when interpreting the results. " \
                 "Consider giving unique names to the models :)")

        self.model1_info['Layers'] = []
        self.model2_info['Layers'] = []

        self.model1_features = {}
        self.model2_features = {}

        if len(list(model1.modules())) > 150 and model1_layers is None:
            warn("Model 1 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model1_layers' parameter. Your CPU/GPU will thank you :)")

        self.model1_layers = model1_layers

        if len(list(model2.modules())) > 150 and model2_layers is None:
            warn("Model 2 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model2_layers' parameter. Your CPU/GPU will thank you :)")

        self.model2_layers = model2_layers

        self._insert_hooks()
        # self.model1 = self.model1.to(self.device)
        # self.model2 = self.model2.to(self.device)

        self.model1.eval()
        self.model2.eval()
    def _log_layer(self,
                   model: str,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):
        with torch.no_grad():
            if model == "model1":
                X = out.flatten(1)
                    
                self.model1_features[name] = (X @ X.t()).fill_diagonal_(0)

            elif model == "model2":
                X = out.flatten(1)
                    
                self.model2_features[name] = (X @ X.t()).fill_diagonal_(0)
                #self.model2_features[name] = out

            else:
                raise RuntimeError("Unknown model name for _log_layer.")

    def _insert_hooks(self):
        # Model 1
        for name, layer in self.model1.named_modules():
            if self.model1_layers is not None:
                if name in self.model1_layers:
                    self.model1_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model1", name))
            else:
                self.model1_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model1", name))

        # Model 2
        for name, layer in self.model2.named_modules():
            if self.model2_layers is not None:
                if name in self.model2_layers:
                    self.model2_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model2", name))
            else:

                self.model2_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model2", name))

    def _BHSIC2(self, K):
        """
        Computes the unbiased estimate of HSIC metric.
        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        with torch.no_grad():
            #print(K.shape)

            N = K.shape[1]
            result = torch.div(torch.pow(torch.sum(torch.sum(K,dim=-2),dim=-1),2),(N - 1) * (N - 2))#or sum(K)/N-1  * sum(K)/N-2
            result = torch.add(result,torch.sum(torch.diagonal(torch.matmul(K,K),dim1=1,dim2=2),dim=1)) 
            result = torch.sub(result,torch.mul(torch.sum(torch.mul(torch.sum(K,dim=-2),torch.sum(K,dim=-1)),dim=-1),2 / (N - 2)))
            return result

    def _BHSIC( self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.
        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        with torch.no_grad():
            N = K.shape[0]

            
            trace = torch.sum(torch.diagonal(torch.matmul(K.unsqueeze(1),L),dim1=-2,dim2=-1),dim=-1) 
            resa=torch.div(torch.sum(torch.sum(K,dim=-2),dim=-1,keepdim=True),N-1)
            resb=torch.sum(torch.sum(L,dim=-2),dim=-1,keepdim=True)
            result= torch.sub(resa@resb.t(),torch.mul(torch.sum(K,dim=-2)@torch.sum(L,dim=-1).t(),2))

            return torch.add(trace,result,alpha=1/(N-2))
    # def _HSIC(self, K, L):
    #     """
    #     Computes the unbiased estimate of HSIC metric.

    #     Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
    #     """
    #     with torch.no_grad():
    #         N = K.shape[0]
    #         comp = (torch.sum(torch.sum(K,dim=1)) * torch.sum(torch.sum(L,dim=1)) / (N - 1)) - ((torch.sum(K,dim=0) @ torch.sum(L,dim=1)) * 2)
    #         result= torch.add(torch.trace(K@L),comp,alpha=1/(N - 2))
    #     return result.item()
    def _orig_HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.
        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        return torch.trace(K@L).add(((torch.sum(K)*torch.sum(L))/(N - 1))-(torch.sum(K,dim=0)@torch.sum(L,dim=1)*2),alpha=1/(N-2))
        #return result.item()
    def compare(self,
                dataloader1: DataLoader,
                dataloader2: DataLoader = None) -> None:
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param dataloader1: (DataLoader)
        :param dataloader2: (DataLoader) If given, model 2 will run on this
                            dataset. (default = None)
        """

        if dataloader2 is None:
            warn("Dataloader for Model 2 is not given. Using the same dataloader for both models.")
            dataloader2 = dataloader1

        self.model1_info['Dataset'] = dataloader1.dataset.__repr__().split('\n')[0]
        self.model2_info['Dataset'] = dataloader2.dataset.__repr__().split('\n')[0]

        N = len(self.model1_layers) if self.model1_layers is not None else len(list(self.model1.modules()))
        M = len(self.model2_layers) if self.model2_layers is not None else len(list(self.model2.modules()))

        self.hsic_matrix0 = torch.zeros(N)
        self.hsic_matrix1 = torch.zeros(N, M)
        self.hsic_matrix2 = torch.zeros(M)

        num_batches = min(len(dataloader1), len(dataloader2))

        for (x1, *_), (x2, *_) in tqdm(zip(dataloader1, dataloader2), desc="| Comparing features |", total=num_batches):

            self.model1_features = {}
            self.model2_features = {}

            x1=x1.to(self.device,non_blocking=True)
            model=self.model1.to(self.device,non_blocking=True)
            model(x1)
            del x1 #free memory
            model=self.model2.to(self.device,non_blocking=True)
            x2=x2.to(self.device,non_blocking=True)
            model(x2)
            del x2,model#free memory

            for i,K in enumerate(self.model1_features.values()):
                self.hsic_matrix0[i] += self._orig_HSIC(K, K).cpu()
            for i,L in enumerate(self.model2_features.values()):
                self.hsic_matrix2[i] += self._orig_HSIC(L, L).cpu()
            for i,K in enumerate(self.model1_features.values()):            
               for j,L in enumerate(self.model2_features.values()):
                    assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"
                    self.hsic_matrix1[i, j] += self._orig_HSIC(K, L).cpu()
                    #self.hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches

        self.hsic_matrix = self.hsic_matrix1 / (self.hsic_matrix0.unsqueeze(1).sqrt() *
                                                        self.hsic_matrix2.unsqueeze(0).sqrt())

        
        if not torch.isnan(self.hsic_matrix).any():
            warn("HSIC computation resulted in NANs")
    def Bcompare(self,
                dataloader1: DataLoader,
                dataloader2: DataLoader = None) -> None:
        if dataloader2 is None:
            warn("Dataloader for Model 2 is not given. Using the same dataloader for both models.")
            dataloader2 = dataloader1

        self.model1_info['Dataset'] = dataloader1.dataset.__repr__().split('\n')[0]
        self.model2_info['Dataset'] = dataloader2.dataset.__repr__().split('\n')[0]

        N = len(self.model1_layers) if self.model1_layers is not None else len(list(self.model1.modules()))
        M = len(self.model2_layers) if self.model2_layers is not None else len(list(self.model2.modules()))

        self.hsic_matrix=torch.zeros(N,M)
        num_batches = min(len(dataloader1), len(dataloader2))

        for (x1, *_), (x2, *_) in tqdm(zip(dataloader1, dataloader2), desc="| Comparing features |", total=num_batches):

            self.model1_features = {}
            self.model2_features = {}

            x1=x1.to(self.device,non_blocking=True)
            model=self.model1.to(self.device,non_blocking=True)
            model(x1)
            del x1 #free memory
            model=self.model2.to(self.device,non_blocking=True)
            x2=x2.to(self.device,non_blocking=True)
            model(x2)
            del x2,model#free memory
            features=list(self.model1_features.values())
            features2=list(self.model2_features.values())
            
            m2_matrix=self._BHSIC2(torch.stack(features2,dim=0)).unsqueeze(0)#//(50 * (50 - 3))
            m0_matrix=self._BHSIC2(torch.stack(features,dim=0)).unsqueeze(1)#/(50 * (50 - 3))
            m1_matrix =self._BHSIC(torch.stack(features,dim=0),torch.stack(features2,dim=0))#/(50 * (50 - 3))
            res_matrix = torch.div(m1_matrix, torch.sqrt(torch.abs(torch.mul(m0_matrix,m2_matrix)))).cpu()
            self.hsic_matrix=torch.add(self.hsic_matrix,res_matrix)
            #self.hsic_matrix = self.hsic_matrix[:, :, 1] / (self.hsic_matrix[:, :, 0].sqrt() *
            #                                                        'self.hsic_matrix[:, :, 2].sqrt())

        if not torch.isnan(self.hsic_matrix).any():
            warn("HSIC computation resulted in NANs")

    def export(self) -> Dict:
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        return {
            "model1_name": self.model1_info['Name'],
            "model2_name": self.model2_info['Name'],
            "CKA": self.hsic_matrix,
            "model1_layers": self.model1_info['Layers'],
            "model2_layers": self.model2_info['Layers'],
            "dataset1_name": self.model1_info['Dataset'],
            "dataset2_name": self.model2_info['Dataset'],

        }

    def plot_results(self,
                     save_path: str = None,
                     title: str = None):
        fig, ax = plt.subplots()
        im = ax.imshow(self.hsic_matrix, origin='lower', cmap='magma')
        ax.set_xlabel(f"Layers {self.model2_info['Name']}", fontsize=15)
        ax.set_ylabel(f"Layers {self.model1_info['Name']}", fontsize=15)

        if title is not None:
            ax.set_title(f"{title}", fontsize=18)
        else:
            ax.set_title(f"{self.model1_info['Name']} vs {self.model2_info['Name']}", fontsize=18)

        add_colorbar(im)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        plt.show()