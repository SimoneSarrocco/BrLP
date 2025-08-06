from datalist import DataList
from pathlib import Path
import yaml

if __name__ == "__main__":
    src_path = Path(__file__).parent
    with open(src_path / '/home/simone.sarrocco/OMEGA_study/BrLP/src/configs/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(src_path / '/home/simone.sarrocco/OMEGA_study/BrLP/src/configs/lakefs_cfg.yaml') as f:
        lakefs_config = yaml.load(f, Loader=yaml.FullLoader)

    dataset = DataList.from_lakefs(
        data_config=config["data"],
        lakefs_config=lakefs_config,
        filepath='data/',
        include_root=True,
        shuffle=True
    )

    print(f"Dataset has {len(dataset.data['training'])} samples")