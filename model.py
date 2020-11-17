import torch.nn as nn
import torchvision
from torchvision.models.resnet import BasicBlock
class BWatcher(nn.Module):
    def __init__(self,stop_block=3,n_classes=20,rpn=True,cfs=0.85,reset_params=True,transform=None):
        super(BWatcher,self).__init__()
        self.eff_net= EfficientNet.from_pretrained('efficientnet-b7')
        self.drop_connect_rate = self.eff_net._global_params.drop_connect_rate
        self.fc=nn.Linear(2560,n_classes)
        self.softm=nn.Softmax()
        self.sb=stop_block
        self.cfs=cfs

        self.transform=transform

        if rpn:
            self.rpn=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            self.rpn.eval()
            self.rescale=transforms.Resize((224,224))
        if reset_params: self.__reset_parameters(stop_block)
    def __reset_parameters(self,init_idx):
        for m in self.eff_net._blocks[init_idx:]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def crop(self,imgs):
        res=[]
        self.rpn.eval()
        with torch.no_grad():
            outputs=self.rpn(imgs)
            for out,img in zip(outputs,imgs):
                box=None
                for b, label,score in zip(out['boxes'],out['labels'],out['scores']):
                    if label==16 and score>self.cfs:
                        box=tuple(int(t) for t in b.cpu())
                        break
                    
                print(img.shape)
                if box:  cropped_img=img[:,box[1]:box[3],box[0]:box[2]]
                else: cropped_img=img
                new_img=self.rescale(cropped_img.unsqueeze(0))
                res.append(new_img)
        return torch.cat(res)
            
    def forward(self,inputs):

        if self.cfs: inputs=self.crop(inputs)
        
        if self.transform: inputs=self.transform(inputs)


        with torch.no_grad():
            x = self.eff_net._swish(self.eff_net._bn0(self.eff_net._conv_stem(inputs)))
            prev_x = x
            for idx, block in enumerate(self.eff_net._blocks[:self.sb]):
                drop_connect_rate=self.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / len(self.eff_net._blocks) # scale drop connect_rate
                x = block(x, drop_connect_rate=drop_connect_rate)

        for idx,block in enumerate(self.eff_net._blocks[self.sb:]):
            idx=idx+self.sb
            drop_connect_rate=self.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.eff_net._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
        x = self.eff_net._swish(self.eff_net._bn1(self.eff_net._conv_head(x)))
        x = self.eff_net._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.eff_net._dropout(x)
        x=self.softm(self.fc(x))
        return x

