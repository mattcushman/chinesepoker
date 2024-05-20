from CPMLTorchEnv import CPMLTorchEnv
from torch import nn
from torchrl.modules.tensordict_module.actors import QValueActor

observation_keys = ["hand", "tomove", "player", "actionhistory"]

def main():
    torchenv = CPMLTorchEnv(seed=1)
    
    model = nn.Sequential(
        nn.Linear(52, 128),
        nn.ReLU(),
        nn.Linear(128, 52),
    )

    actor = QValueActor(model, 
                        in_keys=observation_keys,
                        action_
                        torchenv.specs)


if __name__ == "__main__":
    main()
   