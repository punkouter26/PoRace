"""G1 joystick env = mujoco_playground's stock G1 Joystick on the PoRace-derived model (training/g1/assets/g1_porace.xml).
Rewards, observations, pushes and the gait clock are the stock ones; only the model file differs (see make_model.py)."""
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ml_collections import config_dict
from mujoco_playground._src.locomotion.g1 import base as g1_base
from mujoco_playground._src.locomotion.g1 import joystick as stock

XML = Path(__file__).resolve().parent / "assets" / "g1_porace.xml"
default_config = stock.default_config


class G1Joystick(stock.Joystick):
  def __init__(self, config: config_dict.ConfigDict = None,
               config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None):
    g1_base.G1Env.__init__(self, xml_path=str(XML), config=config or default_config(), config_overrides=config_overrides)
    self._post_init()
