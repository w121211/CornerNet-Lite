from .base import Base, load_cfg, load_nnet
from .paths import get_file_path
from .config import SystemConfig
from .dbs.coco import COCO

class CornerNet(Base):
    def __init__(self):
        from .test.cornernet import cornernet_inference
        from .models.CornerNet import model

        cfg_path   = get_file_path("..", "configs", "CornerNet.json")
        model_path = get_file_path("..", "cache", "nnet", "CornerNet", "CornerNet_500000.pkl")

        cfg_sys, cfg_db = load_cfg(cfg_path)
        sys_cfg = SystemConfig().update_config(cfg_sys)
        coco    = COCO(cfg_db)

        cornernet = load_nnet(sys_cfg, model())
        super(CornerNet, self).__init__(coco, cornernet, cornernet_inference, model=model_path)

class CornerNet_Squeeze(Base):
    def __init__(self):
        from .test.cornernet import cornernet_inference
        from .models.CornerNet_Squeeze import model

        cfg_path   = get_file_path("..", "configs", "CornerNet_Squeeze.json")
        # model_path = get_file_path("..", "cache", "nnet", "CornerNet_Squeeze", "CornerNet_Squeeze_500000.pkl")
        model_path = get_file_path("..", "cache", "nnet", "CornerNet_Squeeze", "CornerNet_Squeeze_405000.pkl")

        cfg_sys, cfg_db = load_cfg(cfg_path)
        sys_cfg = SystemConfig().update_config(cfg_sys)
        coco    = COCO(cfg_db)

        cornernet = load_nnet(sys_cfg, model())
        super(CornerNet_Squeeze, self).__init__(coco, cornernet, cornernet_inference, model=model_path)

class CornerNet_Saccade(Base):
    def __init__(self):
        from .test.cornernet_saccade import cornernet_saccade_inference
        from .models.CornerNet_Saccade import model

        cfg_path   = get_file_path("..", "configs", "CornerNet_Saccade.json")
        model_path = get_file_path("..", "cache", "nnet", "CornerNet_Saccade", "CornerNet_Saccade_500000.pkl")

        cfg_sys, cfg_db = load_cfg(cfg_path)
        sys_cfg = SystemConfig().update_config(cfg_sys)
        coco    = COCO(cfg_db)

        cornernet = load_nnet(sys_cfg, model())
        super(CornerNet_Saccade, self).__init__(coco, cornernet, cornernet_saccade_inference, model=model_path)


class CornerNet_Squeeze_plus(Base):
    def __init__(self, *model_path):
        from .test.cornernet_plus import cornernet_inference
        from .models.CornerNet_Squeeze_plus import model

        cfg_path   = get_file_path("..", "configs", "CornerNet_Squeeze_plus.json")
        # model_path = get_file_path("..", "cache", "nnet", "CornerNet_Squeeze", "CornerNet_Squeeze_500000.pkl")
        # model_path = get_file_path("..", "cache", "nnet", "CornerNet_Squeeze_plus", "CornerNet_Squeeze_plus_500000.pkl")
        model_path = get_file_path(*model_path)

        cfg_sys, cfg_db = load_cfg(cfg_path)
        sys_cfg = SystemConfig().update_config(cfg_sys)
        coco    = COCO(cfg_db)

        cornernet = load_nnet(sys_cfg, model())
        super(CornerNet_Squeeze_plus, self).__init__(coco, cornernet, cornernet_inference, model=model_path)