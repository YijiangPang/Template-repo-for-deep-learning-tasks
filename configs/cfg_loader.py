import sys

def cfg_loader():
    index_cfg = sys.argv[1]
    sys.argv.pop(1)
    if index_cfg == "ImgCla":
        from configs.cfg_ImgCla import cfg_ImgCla
        cfg = cfg_ImgCla()
    elif index_cfg == "LanCla":
        from configs.cfg_LanCla import cfg_LanCla
        cfg = cfg_LanCla()
    elif index_cfg == "CL":
        from configs.cfg_CL import cfg_CL
        cfg = cfg_CL()
    else:
        raise Exception('cfg index does not exist - %s !'%(index_cfg)) 
    
    return cfg.get_cfg(), index_cfg

