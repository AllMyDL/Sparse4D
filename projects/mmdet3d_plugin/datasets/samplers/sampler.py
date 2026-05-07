from mmcv.utils.registry import Registry, build_from_cfg

SAMPLER = Registry("sampler")


def build_sampler(cfg, default_args):
    # 和 dataset/model 一样，sampler 也通过 registry 动态实例化。
    return build_from_cfg(cfg, SAMPLER, default_args)
