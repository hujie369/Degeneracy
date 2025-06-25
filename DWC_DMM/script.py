import os

typeId = [1, 2, 2]
bit = [2, 3, 4]

# DWC and DMM
for i in range(len(bit)):
    for m in [2**i for i in range(0, 10)]:
        command = f"make Custom_Method bit={bit[i]} typeId={typeId[i]} " \
            f"m={m} n=1000 k=512 factor={2**bit[i]} Output=resnet18"
        os.system(command)

# regular method
for i in range(len(bit)):
    for m in [2**i for i in range(0, 10)]:
        command = f"make Regular_Method bit={bit[i]} typeId={typeId[i]} " \
            f"m={m} n=1000 k=512 factor={2**bit[i]} Output=resnet18"
        os.system(command)
