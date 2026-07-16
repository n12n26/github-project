{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9jDS6PWgrbTc"
      },
      "outputs": [],
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "from torchvision import datasets, transforms\n",
        "from torch.utils.data import DataLoader"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "transform = transforms.ToTensor()\n",
        "\n",
        "train_data = datasets.MNIST(root=\"./data\", train=True, download=True, transform=transform)\n",
        "test_data = datasets.MNIST(root=\"./data\", train=False, download=True, transform=transform)\n",
        "\n",
        "train_loader = DataLoader(train_data, batch_size=64, shuffle=True)\n",
        "test_loader = DataLoader(test_data, batch_size=64)"
      ],
      "metadata": {
        "id": "VjxfZlTssDai",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e731ecdd-f03c-4b4f-d875-a74516c85e0d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 9.91M/9.91M [00:00<00:00, 61.3MB/s]\n",
            "100%|██████████| 28.9k/28.9k [00:00<00:00, 1.71MB/s]\n",
            "100%|██████████| 1.65M/1.65M [00:00<00:00, 14.4MB/s]\n",
            "100%|██████████| 4.54k/4.54k [00:00<00:00, 7.24MB/s]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "class SimpleNN(nn.Module):\n",
        "    def __init__(self):\n",
        "        super().__init__()\n",
        "        self.model = nn.Sequential(\n",
        "            nn.Flatten(),          # 28x28 → 784\n",
        "            nn.Linear(784, 128),\n",
        "            nn.ReLU(),\n",
        "            nn.Linear(128, 10)     # 10 classes\n",
        "        )\n",
        "\n",
        "    def forward(self, x):\n",
        "        return self.model(x)"
      ],
      "metadata": {
        "id": "i9oYmx5PsFy-"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "\n",
        "model = SimpleNN().to(device)\n",
        "criterion = nn.CrossEntropyLoss()\n",
        "optimizer = optim.Adam(model.parameters(), lr=0.001)"
      ],
      "metadata": {
        "id": "TdRouy77sO8p"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "for epoch in range(3):\n",
        "    for images, labels in train_loader:\n",
        "        images, labels = images.to(device), labels.to(device)\n",
        "\n",
        "        outputs = model(images)\n",
        "        loss = criterion(outputs, labels)\n",
        "\n",
        "        optimizer.zero_grad()\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "\n",
        "    print(f\"Epoch {epoch+1} done\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1En8jEofsRUm",
        "outputId": "5886f33e-e27e-41b6-b1c5-82db93bc195c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1 done\n",
            "Epoch 2 done\n",
            "Epoch 3 done\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "torch.save(model.state_dict(), \"model.pth\") # save model weights"
      ],
      "metadata": {
        "id": "BYq3TjJkrm8M"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "correct = 0\n",
        "total = 0\n",
        "\n",
        "model.eval()\n",
        "with torch.no_grad():\n",
        "    for images, labels in test_loader:\n",
        "        images, labels = images.to(device), labels.to(device)\n",
        "\n",
        "        outputs = model(images)\n",
        "        preds = torch.argmax(outputs, dim=1)\n",
        "\n",
        "        total += labels.size(0)\n",
        "        correct += (preds == labels).sum().item()\n",
        "\n",
        "print(\"Accuracy:\", 100 * correct / total)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "oyI8ThR8sT3p",
        "outputId": "38e27279-bbdf-4091-abd7-e8bd8d07c43e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: 96.83\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 806
        },
        "id": "eef4993f",
        "outputId": "1b931957-cb09-42b8-cfc9-d7f32e4c9a48"
      },
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Get a batch of test images and labels\n",
        "dataiter = iter(test_loader)\n",
        "images, labels = next(dataiter)\n",
        "\n",
        "# Move to device\n",
        "images, labels = images.to(device), labels.to(device)\n",
        "\n",
        "# Make predictions\n",
        "model.eval()\n",
        "with torch.no_grad():\n",
        "    outputs = model(images)\n",
        "    preds = torch.argmax(outputs, dim=1)\n",
        "\n",
        "# Display some images with predictions and true labels\n",
        "fig = plt.figure(figsize=(10, 8))\n",
        "for i in range(9):\n",
        "    ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])\n",
        "    # Convert image to numpy array for displaying\n",
        "    img = images[i].cpu().numpy().squeeze()\n",
        "    ax.imshow(img, cmap='gray')\n",
        "    ax.set_title(f\"True: {labels[i].item()}\\nPred: {preds[i].item()}\",\n",
        "                 color=(\"green\" if preds[i] == labels[i] else \"red\"))\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x800 with 9 Axes>"
            ],
            "image/png": "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Pu2caUIUsEFK/T7u9km3x3Mznlvz87jp42Lc9HEREbHPlvsIUTWorPDv64gAAADk8p0oAACABEIUAABAAiEKAAAggRAFAACQQIgCAABIUK0tzisqKmL27NnRvn37KCsrq+2eoEErFAqxePHi6N69ezRr5vcUDYE5DqrH/NbwmN+g+lLmuGqFqNmzZ0fPnj1rpDloKt5///3o0aNHqdugGsxxkMb81nCY3yBddea4av0aqX379jXSEDQlXjcNh+cK0njNNByeK0hXnddNtUKU5V9I53XTcHiuII3XTMPhuYJ01Xnd+EAzAABAAiEKAAAggRAFAACQQIgCAABIIEQBAAAkEKIAAAASCFEAAAAJhCgAAIAEQhQAAEACIQoAACCBEAUAAJBAiAIAAEggRAEAACQQogAAABIIUQAAAAmEKAAAgARCFAAAQAIhCgAAIIEQBQAAkKB5qRsAoGH4/ve/n1lr3bp1Zm3HHXfMHffoo48uqp9bbrklt/7SSy9l1u6+++6izgkAEVaiAAAAkghRAAAACYQoAACABEIUAABAAiEKAAAggRAFAACQwBbnAKxx3333ZdaK3Yp8fSoqKoo6buTIkbn1oUOHZtaee+65zNrMmTOL6gegrvTv3z+z9vbbb2fWzjnnnMzajTfeuEE9NTVWogAAABIIUQAAAAmEKAAAgARCFAAAQAIhCgAAIIEQBQAAkMAW5wBNSN4W5hG1s4153na7ERFjx47NrPXt2zez9uUvfzl33H79+mXWTjjhhMzaVVddlTsuQKntsssumbW8y0bMmjWrNtppkqxEAQAAJBCiAAAAEghRAAAACYQoAACABEIUAABAAiEKAAAggRAFAACQwHWiABqZ3XbbLbN2xBFHFD3um2++mVk77LDDMmvz5s3LHXfJkiWZtY033jiz9vLLL+eOu9NOO2XWunTpknssQH228847Z9aWLl2aWXvooYdqoZumyUoUAABAAiEKAAAggRAFAACQQIgCAABIIEQBAAAkEKIAAAASNNotzo8++ujM2umnn55Zmz17du64K1asyKyNHj06s/bhhx9m1t59993ccwKk2HzzzTNrZWVlucfmbWN+0EEHZdbmzJmz/saK8L3vfS+ztt122xU97p/+9KeijwWobQMGDMitn3XWWZm1u+++u6bbYR2sRAEAACQQogAAABIIUQAAAAmEKAAAgARCFAAAQAIhCgAAIEGj3eL86quvzqz17t27Vs45cuTIzNrixYsza3lbCjc2s2bNyqzlPWevvPJKbbQDjdJjjz2WWdtqq61yj82bq+bPn190T8U6/vjjM2stWrSow04A6s7nP//53Hrbtm0za/fdd19Nt8M6WIkCAABIIEQBAAAkEKIAAAASCFEAAAAJhCgAAIAEQhQAAEACIQoAACBBo71O1Omnn55Z23HHHTNrb731Vu642267bWZt1113zawNHjw4s7bnnnvmnvP999/PrPXs2TP32GKtXr06szZ37tzM2uabb170OWfOnJlZc50oqBkzZswodQtrOe+88zJr/fv3L3rcv/zlL0XVAErt/PPPz63nzeXeM9UNK1EAAAAJhCgAAIAEQhQAAEACIQoAACCBEAUAAJBAiAIAAEjQaLc4f+aZZ4qqrc8TTzxR1HGdOnXKrO288865x06cODGztvvuuxfVz/qsWLEis/aPf/wjs7a+LeI7d+6cWZs6der6GwManOHDh+fWL7/88szaxhtvnFn76KOPcse94IILMmvLli3LPRagtvXu3Tuztttuu+Uem/debOnSpcW2RAIrUQAAAAmEKAAAgARCFAAAQAIhCgAAIIEQBQAAkECIAgAASNBotzivbxYsWJBZGzduXNHjbsh27cU66qijMmt5W7lHRLz++uuZtfvuu6/onoD6a31b9eZtY55nfXPGc889V9S4AHVh0KBBRR87d+7cGuyEYliJAgAASCBEAQAAJBCiAAAAEghRAAAACYQoAACABEIUAABAAlucs07dunXLrN18882ZtWbN8nP55ZdfnlmbP3/++hsD6qWHH344s3bggQcWPe5dd92VWbv44ouLHheg1HbYYYeij7366qtrsBOKYSUKAAAggRAFAACQQIgCAABIIEQBAAAkEKIAAAASCFEAAAAJhCgAAIAErhPFOn3rW9/KrHXt2jWztmDBgtxx/+///q/onoDS2nzzzTNre+21V2atZcuWuePOmzcvszZq1KjM2pIlS3LHBSi1PffcM7N2yimnZNYmTZqUO+5TTz1VdE/UDCtRAAAACYQoAACABEIUAABAAiEKAAAggRAFAACQQIgCAABIYIvzJmzvvffOrP3whz8saszDDz88t/7GG28UNS5Qen/4wx8ya126dCl63HvuuSezNnXq1KLHBSi1oUOHZtY6d+6cWXviiSdyx12xYkXRPVEzrEQBAAAkEKIAAAASCFEAAAAJhCgAAIAEQhQAAEACIQoAACCBLc6bsGHDhmXWWrRokVl75plnMmsvvfTSBvUElNZhhx2WWdt1112LGvPZZ5/NrV9yySVFjQtQ3+20006ZtUKhkFl74IEHaqMdapCVKAAAgARCFAAAQAIhCgAAIIEQBQAAkECIAgAASCBEAQAAJBCiAAAAErhOVCPWunXr3PrBBx+cWVu5cmVmLe+aLqtWrVp/Y0DJdOnSJbd+4YUXZtbyrh+X59VXX82tL1mypKhxAUpts802y63vu+++mbX/+7//y6w99NBDRfdE3bASBQAAkECIAgAASCBEAQAAJBCiAAAAEghRAAAACYQoAACABLY4b8TOO++83Pouu+ySWXviiScyay+++GLRPQGl9b3vfS+3vvvuuxc17sMPP5xZy7ssAkBDdvLJJ+fWu3Xrlln785//XMPdUJesRAEAACQQogAAABIIUQAAAAmEKAAAgARCFAAAQAIhCgAAIIEtzhu4Qw89NLP2ox/9KPfYTz75JLN2+eWXF90TUH9997vfrZVxzzrrrMzakiVLauWcAKXWq1evoo9dsGBBDXZCXbMSBQAAkECIAgAASCBEAQAAJBCiAAAAEghRAAAACYQoAACABLY4bwC6dOmSWfvFL36RWdtoo41yx3388cczay+//PL6GwP4/zp37pxZW7VqVR128i+LFi3KrOX11KJFi8xax44di+7nc5/7XGattraeLy8vz6z94Ac/yD122bJlNd0ONDrDhw8v+tjHHnusBjuhrlmJAgAASCBEAQAAJBCiAAAAEghRAAAACYQoAACABEIUAABAAiEKAAAggetE1QPru57TE088kVnr06dPZm3q1Km54/7oRz/Kbwygml577bVSt7CWMWPGZNbmzJmTWdt0000za8cdd9wG9VSffPjhh7n1K664oo46gfptn332yaxtttlmddgJ9YmVKAAAgARCFAAAQAIhCgAAIIEQBQAAkECIAgAASCBEAQAAJLDFeT3Qr1+/3PrAgQOLGve73/1ubn19W6ADjc/jjz+eW//KV75SR53UvmOOOabOz7l69erMWkVFRdHjPvroo5m1V155pagxn3/++WLbgSbliCOOyKyt7zI1kyZNyqyNHz++6J4oPStRAAAACYQoAACA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QAIhCgAAIIEQBQAAkECIAgAASCBEAQAAJBCiAAAAEghRAAAACYQoAACABEIUAABAAiEKAAAggRAFAACQQIgCAABIIEQBAAAkEKIAAAASNC91A0S0bds2t37NNddk1kaOHJlZmzhxYu64xxxzTGZtxowZuccCAEBTZSUKAAAggRAFAACQQIgCAABIIEQBAAAkEKIAAAASCFEAAAAJbHFeD2y++ea59dNPPz2zVlFRkVkbOHBg7rjDhw/PrN100025xwJNz6677ppZe/DBBzNrvXv3roVuSuPAAw/Mrb/11luZtffff7+m2wGaqC9/+cuZtUcffTSzdtZZZ2XWbr311txzlpeXr7+xJsRKFAAAQAIhCgAAIIEQBQAAkECIAgAASCBEAQAAJBCiAAAAEghRAAAACVwnqo507do1s3bnnXfWYScAxTnooIMyay1btqzDTkon79osERGnnnpqZu3444+v6XaARqpLly659ZtvvrmocX/5y19m1n7729/mHrt8+fKiztlYWYkCAABIIEQBAAAkEKIAAAASCFEAAAAJhCgAAIAEQhQAAEACW5zXoG9/+9uZtcMPPzyztscee9RCN+u33377ZdaaNcvO15MnT86sjR8/foN6AkqrefPs/y0MGzasDjupnyZOnJhb/+53v5tZa9u2bWZt6dKlRfcEND5579EiInr06FHUuPfee29mbcWKFUWN2VRZiQIAAEggRAEAACQQogAAABIIUQAAAAmEKAAAgARCFAAAQAJbnNegn//855m1ioqKOuykeo488siiajNmzMisHXfccbnnXN/2wEBpDRkyJLP2xS9+MbN29dVX10Y79U6nTp1y69ttt11mrU2bNpk1W5xD09OyZcvM2kUXXVQr57z77rsza4VCoVbO2VhZiQIAAEggRAEAACQQogAAABIIUQAAAAmEKAAAgARCFAAAQAIhCgAAIIHrRCV6/PHHM2vNmtWvTPrxxx/n1pcsWZJZ69WrV2atT58+mbW//vWvuefcaKONcutA7RowYEBu/d57782sTZ06NbN25ZVXFt1TQ/KVr3yl1C0AjcQOO+yQWRs4cGDR465evTqz9uc//7nocamqfr3rBwAAqOeEKAAAgARCFAAAQAIhCgAAIIEQBQAAkECIAgAASGCL8/8waNCg3Po222yTWauoqCiqtiFuvfXWzNqTTz6Ze+yiRYsya1/60pcyaxdddNH6G8vwzW9+M7N2yy23FD0uUD0XX3xxbr1t27aZtYMPPjizlnfJhIamc+fOmbX1/T+ituZ6oPE56qijamXc9b3/o2ZYiQIAAEggRAEAACQQogAAABIIUQAAAAmEKAAAgARCFAAAQIImucV57969M2v/8z//k3vsJptsUsPdRMyYMSO3/oc//CGzdtlll2XWli1bVis9feMb38isde3aNXfcq6++OrPWqlWrzNovf/nL3HFXrVqVW4em5Oijj86sDRs2LPfYd999N7P2yiuvFN1TQ5J3GYf1bWH+7LPPZtYWLlxYZEdAY7TffvsVfezKlSszaxtyKRqqz0oUAABAAiEKAAAggRAFAACQQIgCAABIIEQBAAAkEKIAAAASNMktzps3z37YtbGFeUTEc889l1k7/vjjc4+dN29eTbezXnlbnF911VWZteuuuy533DZt2mTW8rY/f/TRR3PHnTp1am4dmpJjjjkms5b3GoyIuPnmm2u6nXop71IXJ5xwQmatvLw8d9xRo0Zl1lyKAZqevfbaq6ja+ixdujSz9uqrrxY9LtVnJQoAACCBEAUAAJBAiAIAAEggRAEAACQQogAAABIIUQAAAAmEKAAAgARN8jpRteWVV17JrJ166qmZtVJcB2pD5F2zKe/6KhERu+++e023A01Sx44dM2t77rln0ePecsstRR/bkHzjG9/IrOVdL/Ctt97KHXfcuHFF9wQ0PrX1vqepzNX1mZUoAACABEIUAABAAiEKAAAggRAFAACQQIgCAABIIEQBAAAksMX5f2jWrPhc+YUvfKEGO6m/ysrKMmvr+/sr9u/30ksvza2fdNJJRY0LDVXLli0za1tssUVm7d57762Ndhqcfv36FXXcG2+8UcOdAI3ZbrvtVtRxCxcuzK3b4rz0rEQBAAAkEKIAAAASCFEAAAAJhCgAAIAEQhQAAEACIQoAACBBk9zi/IwzzsisVVRU1GEnDdOXv/zlzNouu+ySe2ze329ebX1bnENTs3jx4szaq6++mlnbcccdc8ft3LlzZm3+/Pnr7as+6datW2bt6KOPLmrMCRMmFNsO0Ajts88+ufURI0YUNe6iRYty67NmzSpqXGqOlSgAAIAEQhQAAEACIQoAACCBEAUAAJBAiAIAAEggRAEAACQQogAAABI0yetE5V3nqCnp2rVrZm277bbLrF144YW10U7MnTs3s7Zq1apaOSc0VMuXL8+sTZ06NbN21FFH5Y77pz/9KbN23XXXrb+xGjZgwIDMWt++fXOP7d27d2atUCgU1Y9rCQL/rkuXLrn1Zs2KW6946qmnijqOumMlCgAAIIEQBQAAkECIAgAASCBEAQAAJBCiAAAAEghRAAAACZrkFudUuuiiizJr3/rWt2rlnNOnT8+sff3rX8+szZw5sxa6gcbpkksuyayVlZXlHnvooYdm1u69996ieyrWvHnzMmvr26Z8k002qel24o477qjxMYGG6+ijjy762IULF2bWfvWrXxU9LnXDShQAAEACIQoAACCBEAUAAJBAiAIAAEggRAEAACQQogAAABLY4rwRe/zxx3Pr22yzTR118i9TpkzJrE2YMKEOO4HG6+23386sHXvssbnH7rzzzpm1rbbaqtiWivbAAw8Ufeydd96ZWTvhhBOKGnP58uXFtgM0UD169MisjRgxouhxZ82alVl75ZVXih6XumElCgAAIIEQBQAAkECIAgAASCBEAQAAJBCiAAAAEghRAAAACZrkFudlZWWZtWbNis+VhxxySFHH3Xbbbbn17t27FzXu+h5LRUVFUeNuiC9/+ct1fk6g+l599dWiavXRe++9V+NjDhgwILf+xhtv1Pg5gdLaa6+9Mmsb8r7x4YcfLvpYSs9KFAAAQAIhCgAAIIEQBQAAkECIAgAASCBEAQAAJBCiAAAAEghRAAAACZrkdaJuueWWzNrVV19d9Lh//OMfM2sbck2m2rqeU22Me+utt9b4mADFyLsmYF4tj+tAQdPTpUuXoo+dN29eZu2GG24oelxKz0oUAABAAiEKAAAggRAFAACQQIgCAABIIEQBAAAkEKIAAAASNMktzh988MHM2nnnnZd7bNeuXWu6nZKZO3duZu2tt97KrH3jG9/IrM2ZM2eDegKoKYVCoagawL876KCDij525syZmbVFixYVPS6lZyUKAAAggRAFAACQQIgCAABIIEQBAAAkEKIAAAASCFEAAAAJmuQW5zNmzMisHX/88bnHHn744Zm1c845p9iWSuKKK67IrN1000112AlAzWvVqlVRxy1fvryGOwHquxYtWmTW+vXrV/S4K1asyKytWrWq6HEpPStRAAAACYQoAACABEIUAABAAiEKAAAggRAFAACQQIgCAABIIEQBAAAkaJLXicozfvz4outPPvlkZu0b3/hGZu3LX/5y7jkfffTRzNptt92WWSsrK8sdd8qUKbl1gIbslFNOyawtXLgws/aTn/ykFroB6rOKiorM2iuvvJJZGzBgQO647777btE9Ub9ZiQIAAEggRAEAACQQogAAABIIUQAAAAmEKAAAgARCFAAAQAJbnNegJ554oqgaADXvb3/7W2btuuuuy6yNGzeuNtoB6rHy8vLM2kUXXZRZKxQKueNOnDix6J6o36xEAQAAJBCiAAAAEghRAAAACYQoAACABEIUAABAAiEKAAAgQVlhfXszRsQnn3wSHTt2rIt+oNFYtGhRdOjQodRtUA3mOEhjfms4zG+QrjpznJUoAACABEIUAABAAiEKAAAggRAFAACQQIgCAABIIEQBAAAkEKIAAAASCFEAAAAJhCgAAIAEQhQAAEACIQoAACCBEAUAAJBAiAIAAEggRAEAACQQogAAABIIUQAAAAmEKAAAgARCFAAAQAIhCgAAIIEQBQAAkKBaIapQKNR2H9DoeN00HJ4rSOM103B4riBddV431QpRixcv3uBmoKnxumk4PFeQxmum4fBcQbrqvG7KCtWIWhUVFTF79uxo3759lJWV1Uhz0FgVCoVYvHhxdO/ePZo184nZhsAcB9Vjfmt4zG9QfSlzXLVCFAAAAJX8GgkAACCBEAUAAJBAiAIAAEggRAEAACQQogAAABIIUQAAAAmEKAAAgARCFAAAQAIhCgAAIIEQBQAAkECIAgAASCBEAQAAJBCiAAAAEghRAAAACZqXuoHGpOyysmrdb9zXx8Xg3oNrt5kNNHX+1Nj+5u3j0/JP42+n/y12675bqVsCSqyhz3FLVi6Ji//34nhgygMxd9nc6Nupb3x7j2/HN3f/ZqlbA+qDsurNcTFuXMTgwbXaSlF6946YMWPt20eOjLj11jpvp7ETomrQ3UfcXeXnuybfFU+999Rat2+7ybZ12VZRvjP2O9G8WfP4tPzTUrcC1BMNeY4rryiPg+45KF6Z/Up8a/dvxdadt46xU8fGmY+fGQtWLIgL972w1C0CpXZ31bks7ror4qmn1r592/o3x62x884R3/te1dv69y9JK41dWaFQKJS6icbqrMfPipv+dlMULsn/K162alm0adGmjrpav7Hvjo3D/uewOH+v82PU86OsRAHr1JDmuDFvjoljHzg2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2q23Vn6s7le/irjwwsoNKHr3jjjxxMqP+X1m1KjKi/Heemvlrn1f+ELlxXIP3cD3cRdfXLk5xY03Vu729+/BihpVVvj3z0oAAACQy3eiAAAAEghRAAAACYQoAACABEIUAABAAiEKAAAgQbW2OK+oqIjZs2dH+/bto6ysrLZ7ggatUCjE4sWLo3v37tGsmd9TNATmOKge81vDY36D6kuZ46oVombPnh09e/askeagqXj//fejR48epW6DajDHQRrzW8NhfoN01ZnjqvVrpPbtXYQQUnndNByeK0jjNdNweK4gXXVeN9UKUZZ/IZ3XTcPhuYI0XjMNh+cK0lXndeMDzQAAAAmEKAAAgARCFAAAQAIhCgAAIIEQBQAAkECIAgAASCBEAQAAJBCiAAAAEghRAAAACYQoAACABEIUAABAAiEKAAAggRAFAACQQIgCAABIIEQBAAAkEKIAAAASCFEAAAAJhCgAAIAEQhQAAEACIQoAACCBEAUAAJBAiAIAAEggRAEAACQQogAAABIIUQAAAAmEKAAAgATNS90AAABQMzp16pRZ23LLLWvlnDNmzMisfec738k99o033sis/eMf/8isTZ48ef2N1SIrUQAAAAmEKAAAgARCFAAAQAIhCgAAIIEQBQAAkECIAgAASGCL80TdunXLrN1///2ZtRdffDGzdtttt+Wec/r06evtqzHo2LFjZm2//fbLPfaJJ57IrK1atarongAA6tqhhx6aWz/ssMMya4MHD86sbbXVVsW2lCtvK/JevXrlHtuyZcuizrnRRhsVdVxNsRIFAACQQIgCAABIIEQBAAAkEKIAAAASCFEAAAAJhCgAAIAEQhQAAEAC14n6D506dcqtv/nmm5m1vOsc/fOf/8ysNZXrQEXk/x1NnDgxs9a1a9fccQcOHJhZe/fdd9ffGLBBOnTokFm76qqrco8dMGBAZm3o0KGZNdeAA0qtX79+ufVvfetbmbXTTz89s9a6devcccvKyvIbq2P9+/cvdQt1zkoUAABAAiEKAAAggRAFAACQQIgCAABIIEQBAAAkEKIAAAASNMktzjfZZJPM2n333Zd7bOfOnTNrN998c2bt7LPPXn9jTcDFF1+cWevTp09mbeTIkbnj2sYcat8JJ5yQWbviiisyaz179iz6nHlbp3/88cdFjwtQE3r06JFbP+ecc+qok9r39ttvZ9byLgHUWFmJAgAASCBEAQAAJBCiAAAAEghRAAAACYQoAACABEIUAABAgia5xfmuu+6aWRs8eHDR415++eVFH9tYbL/99rn1733ve5m1hx56KLO2vq3ngZqRt13v9ddfn1nr0qVLZq1QKBTdz4033phZO+uss3KPnT9/ftHnBRqevEvYRORvN/7CCy9k1p544onM2qeffpp7zkWLFmXWli5dmllr27Zt7rhPPvlkZu2NN97IrP3lL3/JrE2aNCn3nMuXL8+s5T2WxspKFAAAQAIhCgAAIIEQBQAAkECIAgAASCBEAQAAJBCiAAAAEghRAAAACRrtdaK6deuWWTvqqKOKHve//uu/Mmtz584tetyGJO9aUE8//XTR4+ZdJ2rx4sVFjwtU3/e///3MWufOneuwk0rHHXdcZu3ggw/OPfaKK67IrOVdf2rlypXrbwwoibzrJ+VdOykiYqeddsqsHXHEEUX18/LLL+fW865NOn369MzalltumTvurFmzMmsVFRW5x1IzrEQBAAAkEKIAAAASCFEAAAAJhCgAAIAEQhQAAEACIQoAACBBo93i/L//+78zayeeeGJmbeLEibnjjhkzpuieGot99903s7bpppvmHnvHHXdk1u65555iWwKqqVevXrn1U045pahxX3vttczaP//5z9xjhw4dWtQ5O3bsmFvP26599OjRmbUPP/ywqH6AmrHxxhtn1n7/+99n1vK2MI+IuPLKKzNrG3KJljx525jnmTlzZs02Qo2zEgUAAJBAiAIAAEggRAEAACQQogAAABIIUQAAAAmEKAAAgASNdovzQqGQWauoqMiszZ49O3fclStXFt1TfdO6devM2oUXXphZO/PMMzNreX/vERGnnnrq+hsDas3OO++cW2/fvn1m7fnnn8+sDRo0KLPWqlWr3HN+9atfzazlzUX9+vXLHXezzTbLrD3yyCOZtUMOOSSzNn/+/NxzAuvXrl273PoFF1yQWRs+fHhmbd68ebnjXnvttZm1ZcuW5R4L/8lKFAAAQAIhCgAAIIEQBQAAkECIAgAASCBEAQAAJBCiAAAAEjTaLc6Ldeihh+bWn3zyyczawoULM2u33HJLsS0VLW/L4YiIwYMHZ9b23HPPos75wAMPFHUcUDdatmyZW8+7TMHPf/7zos65YsWK3Prvfve7zNoxxxyTWevbt29R/UTkb2fcmC5lAfXR4Ycfnlv/4Q9/mFmbOXNmZm3ffffNHXfRokW5dUhhJQoAACCBEAUAAJBAiAIAAEggRAEAACQQogAAABIIUQAAAAmEKAAAgASN9jpRN9xwQ2ZtyJAhmbXu3bvnjrvffvtl1srKyjJrhx12WO64tSGvn4j868Hkee+99zJrF154YVFjAnXjq1/9atHH5l1H7+GHHy563Dy77bZbrYz78ssvZ9aWLFlSK+cEKu21115FHztp0qTM2qxZs4oeF1JZiQIAAEggRAEAACQQogAAABIIUQAAAAmEKAAAgARCFAAAQIJGu8X5xIkTM2s77rhjZm3nnXfOHffggw/OrJ133nmZtblz52bW7rzzztxzFuvuu+/OrU+ePLmocV988cXM2tSpU4saE6gb9957b24973IMu+++e2bt85//fGZthx12yD3nEUcckVnr1KlTZm3hwoW54+Yde/rpp2fW8ubOKVOm5J4TWL+jjz666GPz3oddcsklucc+8sgjmbVXX3212JZooqxEAQAAJBCiAAAAEghRAAAACYQoAACABEIUAABAAiEKAAAgQVmhUCis706ffPJJdOzYsS76oQb17ds3t/7uu+9m1vK2+jzooIMya3lbuTc1ixYtig4dOpS6DaqhKc1xnTt3zq3nzQt5f0dlZWWZtWr8bybT008/nVn71re+lXvsH//4x8za1ltvnVn79a9/nVk744wzcs/ZVJjfGo76OL+tb06oqKiolfPmjXvrrbdm1l5++eXM2pZbbpl7zrw59c0338w9Ns/222+fWXvppZcya7NmzSr6nE1JdeY4K1EAAAAJhCgAAIAEQhQAAEACIQoAACCBEAUAAJBAiAIAAEggRAEAACRoXuoGqD0//vGPc+t512n4wQ9+kFlzLShouObPn59bP/bYYzNrDzzwQGZtQ65Dc+ONN2bW8uaiFStW5I774IMPZtZ++MMfZtbyroXXr1+/3HNOnTo1tw5EXHvttbn17373u7Vy3mbNstcOzjzzzKJq9VHe+7Rnn302s3b88cfXQjeNl5UoAACABEIUAABAAiEKAAAggRAFAACQQIgCAABIIEQBAAAkKCvk7XP9/33yyScbtH0tteeYY47JrN133325xy5evDizNmTIkMza3//+9/U3RixatCg6dOhQ6jaoBnNc9QwdOjSzNmLEiMzawoULc8fNuxzDkiVL1ttXltatW2fWfv/732fWDjvssMzaPffck3vOr3/96+tvrBEwvzUc9XF+22ijjXLru+yyS2Yt77XbvHn+lXt69uyZWcvb/rwxyXvbf+mll+YeO2rUqBrupv6qzhzXNP6LAQAAqCFCFAAAQAIhCgAAIIEQBQAAkECIAgAASCBEAQAAJMjfC5J675BDDin62D/+8Y+ZNduYA//p6aefLqpWKsuXL8+s5V0CIm+L87zLP0REdO7cObM2f/783GOhqSgvL8+tv/LKK5m1/v37F33e/fffP7PWokWLzFre1t+777570f2UQllZWWZt4MCBddhJw2clCgAAIIEQBQAAkECIAgAASCBEAQAAJBCiAAAAEghRAAAACWxx3sDlbXG+dOnS3GP/+7//u6bbAWgQ7r///sxa3hbnxx13XO64Z511Vmbt8ssvX39jQK155plnijpu5513zqytb4vz1atXZ9Z+97vfZdZ+/etf54577rnnZtZGjBiReyw1w0oUAABAAiEKAAAggRAFAACQQIgCAABIIEQBAAAkEKIAAAASCFEAAAAJygqFQmF9d/rkk0+iY8eOddEP63DGGWdk1m6++ebM2kcffZQ77mabbVZ0T6zfokWLokOHDqVug2owx/Hv8q4J88ILL+Qe26pVq8zatttum1n7xz/+sd6+6hPzW8Nhfttwu+66a2btb3/7W62cc9y4cbn1wYMHZ9bKysqKOmfee8qIiLPPPruocRui6sxxVqIAAAASCFEAAAAJhCgAAIAEQhQAAEACIQoAACCBEAUAAJCgeakbYP3ytjjP26H+T3/6U9HnbN++fWatU6dOmbWZM2cWfU6A+uDVV1/NrP34xz/OPfaaa67JrF155ZWZtZNOOil33OXLl+fWgdrz1ltvZdbuv//+3GOPPfbYos45ZMiQoo6LiCgvL8+s5b03/OEPf1j0OZsiK1EAAAAJhCgAAIAEQhQAAEACIQoAACCBEAUAAJBAiAIAAEhgi/NGLG+Ly4iIE044IbP2ne98J7P25ptvZta+/vWvr78xgAbqrrvuyq2PHDkys3bkkUdm1i6//PLccV977bX8xoBak3eJgXPPPTf32Hbt2mXWdtttt8xat27dcsedPn16Zu3uu+/OrF166aW541J9VqIAAAASCFEAAAAJhCgAAIAEQhQAAEACIQoAACCBEAUAAJBAiAIAAEhQVigUCuu70yeffBIdO3asi35Yh1dffTWztsMOO2TWysrKcsfNe+pvv/32zNpPfvKTzNr777+fe86mZNGiRdGhQ4dSt0E1mOOoKVtuuWVmLe+6Lvfee2/uuHnX9SsF81vDYX6rv0466aTM2p577pl77GWXXZZZ++ijj4ruiUrVmeOsRAEAACQQogAAABIIUQAAAAmEKAAAgARCFAAAQAIhCgAAIIEtzhuAffbZJ7N2+eWXZ9bGjx+fO+4tt9ySWVuwYEFmbeXKlbnjUskWwA2HOY668OSTT2bWvvjFL+Ye+4UvfCGzNmXKlKJ7Kpb5reEwv0E6W5wDAADUMCEKAAAggRAFAACQQIgCAABIIEQBAAAkEKIAAAASNC91A6zfhAkTMmtf+tKX6rATAIp19NFHZ9YmT56ce+xWW22VWSvFFucATZ2VKAAAgARCFAAAQAIhCgAAIIEQBQAAkECIAgAASCBEAQAAJLDFOQDUgU8++SSz1qdPnzrsBIANZSUKAAAggRAFAACQQIgCAABIIEQBAAAkEKIAAAASCFEAAAAJhCgAAIAEQhQAAEACIQoAACCBEAUAAJBAiAIAAEggRAEAACQQogAAABJUK0QVCoXa7gMaHa+bhsNzBWm8ZhoOzxWkq87rplohavHixRvcDDQ1XjcNh+cK0njNNByeK0hXnddNWaEaUauioiJmz54d7du3j7KyshppDhqrQqEQixcvju7du0ezZj4x2xCY46B6zG8Nj/kNqi9ljqtWiAIAAKCSXyMBAAAkEKIAAAASCFEAAAAJhCgAAIAEQhQAAEACIQoAACCBEAUAAJDg/wEkNc8yfIfoPgAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "OW1PFq2RsVdS"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}