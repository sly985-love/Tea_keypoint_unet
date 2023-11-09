

import torch
from models import unet

# cudnn.benchmark = True


model = unet.U_net().to("cuda:0")
# model.load_state_dict(torch.load("weights/U_net_tea/U_net_460.pth")) weights/U_net_tea/U_net_480.pth
model.load_state_dict(torch.load("weights/U_net_tea/U_net_480.pth"))
model.eval()

example = torch.rand(1, 3, 256, 256).to("cuda:0")  #ZP (300,300) #JG (512,512)
traced_script_module = torch.jit.trace(model, example)

traced_script_module.save("weights/U_net_tea/U_net_480.pt")
