from .CE import CE 
from .Unet import Unet
from .SalPreNet import SalPreNet
from .Transunet import Transunet

model = {
    'CE' : CE, 
    'Unet' : Unet, 
    'SalPreNet' : SalPreNet, 
    'Transunet' : Transunet
}

def get_model(config):
    
    try:
        model_name = config['model']
    except:
        print(f"can not find model:{model_name} defination")
        raise ValueError
    
    print(f'using model:{model_name}')
    
    return model[model_name](config[model_name])
