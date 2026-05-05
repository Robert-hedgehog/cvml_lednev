from encoder_decoder_train import (Decoder, Encoder, ImageDataset)

import torch
import matplotlib.pyplot as plt

encoder = Encoder()
decoder = Decoder()

for option in range(1, 5):
    encoder.load_state_dict(torch.load(f"encoder{option}.pth"))
    decoder.load_state_dict(torch.load(f"decoder{option}.pth"))

    dataset = ImageDataset(10, 256, option)
    image, _ = dataset[0]

    with torch.no_grad():

        latent = encoder(image.unsqueeze(0))
        result = decoder(latent)

        plt.subplot(131)
        plt.imshow(image.squeeze().cpu().numpy())
        plt.subplot(132)
        plt.imshow(result.squeeze().cpu().detach().numpy())
        plt.subplot(133)
        plt.imshow(image.squeeze(0) - result.squeeze())
        plt.show()