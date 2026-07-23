import os

def get_kwargs(meas_path: str, config: dict):
        config = config['measurements']
        for key in meas_path.split(os.sep):
             config = config.get(key, {})
        kwargs = config or dict()
        return kwargs

def iter_bundle(bundle_name, config):
    """Get the measurement paths and kwargs"""
    rel_paths = config['bundles'][bundle_name]
    for meas_path in rel_paths:
        kwargs = get_kwargs(meas_path, config)
        yield meas_path, kwargs

def iter_measurements(config: dict):
    """Get the measurement paths and kwargs"""
    for date_dir, exp_dirs in config['measurements'].items():
        for exp_dir, meas_dirs in exp_dirs.items():
            for meas_dir, kwargs in meas_dirs.items():
                rel_path = os.path.join(date_dir, exp_dir, meas_dir)
                kwargs = kwargs or {}
                yield rel_path, kwargs

def iter_dataset(dataset_name, config: dict):
    """Get the bundle names"""
    yield from config['datasets'][dataset_name]



